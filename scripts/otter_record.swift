// otter-record: capture system audio (the other call participants)
// plus the microphone (the user) into two wav files.
//
// Usage: otter-record <output-dir> [mic-only]
// Writes <output-dir>/system.wav and <output-dir>/mic.wav, prints
// STARTED when both taps are live, and finalizes cleanly on SIGINT or
// SIGTERM. Prints "ERROR <message>" and exits nonzero on failure (a
// missing Screen Recording or Microphone permission surfaces here).
// With "mic-only" (in-person capture) system audio is never touched:
// only mic.wav is written, no Screen Recording permission is needed,
// and LEVEL lines carry only the mic field.
//
// Uses only Apple frameworks: ScreenCaptureKit for system audio (the
// supported, extension-free route) and AVFoundation for the mic.

import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

nonisolated(unsafe) var outDir = URL(fileURLWithPath: ".")

// Latest input levels (0..1 RMS) for each source, so the app can show
// live bars and warn about a dead source. Approximate metering; a lock
// keeps reads and writes from tearing across threads.
final class Levels: @unchecked Sendable {
    private let lock = NSLock()
    private var mic: Float = 0
    private var system: Float = 0

    func setMic(_ v: Float) { lock.lock(); mic = v; lock.unlock() }
    func setSystem(_ v: Float) { lock.lock(); system = v; lock.unlock() }
    func snapshot() -> (Float, Float) {
        lock.lock(); defer { lock.unlock() }; return (mic, system)
    }
}

nonisolated(unsafe) let levels = Levels()

func rms(_ buffer: AVAudioPCMBuffer) -> Float {
    guard let data = buffer.floatChannelData else { return 0 }
    let frames = Int(buffer.frameLength)
    if frames == 0 { return 0 }
    var sum: Float = 0
    let ch = data[0]
    for i in 0..<frames { sum += ch[i] * ch[i] }
    return (sum / Float(frames)).squareRoot()
}

final class SystemAudioWriter: NSObject, SCStreamOutput, SCStreamDelegate {
    var file: AVAudioFile?
    let url: URL

    init(url: URL) { self.url = url }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sb.isValid else { return }
        guard let fmtDesc = CMSampleBufferGetFormatDescription(sb),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(
                  fmtDesc),
              let format = AVAudioFormat(streamDescription: asbd) else {
            return
        }
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sb))
        guard frames > 0,
              let pcm = AVAudioPCMBuffer(pcmFormat: format,
                                         frameCapacity: frames) else {
            return
        }
        pcm.frameLength = frames
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sb, at: 0, frameCount: Int32(frames),
            into: pcm.mutableAudioBufferList)
        guard status == noErr else { return }
        levels.setSystem(rms(pcm))
        do {
            if file == nil {
                file = try AVAudioFile(forWriting: url,
                                       settings: format.settings)
            }
            try file?.write(from: pcm)
        } catch {
            FileHandle.standardError.write(
                "system write failed: \(error)\n".data(using: .utf8)!)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        print("ERROR system audio stream stopped: \(error.localizedDescription)")
        exit(1)
    }
}

final class Recorder {
    let engine = AVAudioEngine()
    var micFile: AVAudioFile?
    var stream: SCStream?
    let systemWriter: SystemAudioWriter

    init() {
        systemWriter = SystemAudioWriter(
            url: outDir.appendingPathComponent("system.wav"))
    }

    func startMic() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        micFile = try AVAudioFile(
            forWriting: outDir.appendingPathComponent("mic.wav"),
            settings: format.settings)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) {
            [weak self] buffer, _ in
            levels.setMic(rms(buffer))
            try? self?.micFile?.write(from: buffer)
        }
        try engine.start()
    }

    func startSystem() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw NSError(domain: "otter", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "no display found"])
        }
        let filter = SCContentFilter(display: display,
                                     excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        // Video is required by the API but unused: minimal size, 1 fps.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let stream = SCStream(filter: filter, configuration: config,
                              delegate: systemWriter)
        try stream.addStreamOutput(systemWriter, type: .audio,
                                   sampleHandlerQueue: DispatchQueue(
                                       label: "otter.audio"))
        try await stream.startCapture()
        self.stream = stream
    }

    func stopAndExit() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        micFile = nil            // closes the file
        let done = DispatchSemaphore(value: 0)
        if let s = stream {
            s.stopCapture { _ in done.signal() }
            _ = done.wait(timeout: .now() + 5)
        }
        systemWriter.file = nil  // closes the file
        print("STOPPED")
        exit(0)
    }
}

@main
struct OtterRecord {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count == 2 || (args.count == 3 && args[2] == "mic-only")
        else {
            print("ERROR usage: otter-record <output-dir> [mic-only]")
            exit(2)
        }
        let micOnly = args.count == 3
        outDir = URL(fileURLWithPath: args[1], isDirectory: true)
        try? FileManager.default.createDirectory(
            at: outDir, withIntermediateDirectories: true)

        let recorder = Recorder()

        // Microphone permission first: this may show the one-time prompt.
        let micOK = await withCheckedContinuation { cont in
            AVCaptureDevice.requestAccess(for: .audio) {
                cont.resume(returning: $0)
            }
        }
        guard micOK else {
            print("ERROR microphone permission denied. Allow it in System "
                  + "Settings > Privacy & Security > Microphone, then try "
                  + "again.")
            exit(1)
        }

        do {
            try recorder.startMic()
        } catch {
            print("ERROR could not start microphone: "
                  + "\(error.localizedDescription)")
            exit(1)
        }

        if !micOnly {
            do {
                // This call needs the system-audio (Screen Recording)
                // permission; the OS shows its prompt the first time.
                try await recorder.startSystem()
            } catch {
                print("ERROR could not capture system audio: "
                      + "\(error.localizedDescription). Allow this app in "
                      + "System Settings > Privacy & Security > Screen & "
                      + "System Audio Recording, then try again.")
                exit(1)
            }
        }

        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        // A dedicated queue: dispatch_main()/main-queue tricks are not
        // legal from a Swift async entry point (SIGTRAP).
        let sigQueue = DispatchQueue(label: "otter.signals")
        let sigint = DispatchSource.makeSignalSource(signal: SIGINT,
                                                     queue: sigQueue)
        sigint.setEventHandler { recorder.stopAndExit() }
        sigint.resume()
        let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM,
                                                      queue: sigQueue)
        sigterm.setEventHandler { recorder.stopAndExit() }
        sigterm.resume()

        print("STARTED")
        // Unbuffered so the Python side sees STARTED immediately.
        fflush(stdout)

        // Emit input levels ~10x/second for the app's meters.
        let meter = DispatchQueue(label: "otter.meter")
        let ticker = DispatchSource.makeTimerSource(queue: meter)
        ticker.schedule(deadline: .now(), repeating: .milliseconds(100))
        ticker.setEventHandler {
            let (mic, system) = levels.snapshot()
            if micOnly {
                print(String(format: "LEVEL mic=%.4f", mic))
            } else {
                print(String(format: "LEVEL mic=%.4f system=%.4f",
                             mic, system))
            }
            fflush(stdout)
        }
        ticker.resume()

        while true {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }
}
