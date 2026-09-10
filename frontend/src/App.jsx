import React, { useEffect, useMemo, useRef, useState } from 'react'

// The one place the display name lives.
const APP_NAME = 'Echo AI'

// The brand mark, inlined so its tile/wave colors follow the app theme
// (the toggle can override the system, and the mark mirrors that).
function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 72 72" aria-hidden="true">
      <rect x="2" y="2" width="68" height="68" rx="16" className="logo-tile" />
      <g className="logo-wave" strokeWidth="5" strokeLinecap="round" fill="none">
        <line x1="16" y1="24" x2="16" y2="48" />
        <line x1="27" y1="18" x2="27" y2="54" />
        <line x1="38" y1="26" x2="38" y2="46" opacity="0.7" />
        <line x1="48" y1="30" x2="48" y2="42" opacity="0.45" />
        <line x1="57" y1="33" x2="57" y2="39" opacity="0.25" />
      </g>
    </svg>
  )
}

function Brand() {
  return (
    <span className="brand" title={APP_NAME}>
      <BrandMark />
      <span className="wordmark">Echo<span className="ai">ai</span></span>
    </span>
  )
}

// Theme: Auto follows the system (no data-theme attribute); Light and
// Dark pin it explicitly. The choice persists in localStorage.
function applyTheme(theme) {
  const root = document.documentElement
  if (theme === 'light' || theme === 'dark') root.dataset.theme = theme
  else delete root.dataset.theme
}

const THEME_MODES = ['light', 'dark', 'auto']

function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem('echoTheme')
    return THEME_MODES.includes(stored) ? stored : 'auto'
  })
  useEffect(() => { applyTheme(theme) }, [theme])
  const pick = (t) => {
    localStorage.setItem('echoTheme', t)
    setTheme(t)
  }
  return (
    <div className="theme-seg" role="group" aria-label="Theme">
      {THEME_MODES.map((t) => (
        <button key={t}
          aria-pressed={theme === t}
          className={theme === t ? 'active' : ''}
          title={t === 'auto' ? 'Follow the system appearance'
            : `Always ${t}`}
          onClick={() => pick(t)}>
          {t[0].toUpperCase() + t.slice(1)}
        </button>
      ))}
    </div>
  )
}

function formatDuration(seconds) {
  if (seconds == null) return ''
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(String(iso).replace(' ', 'T') + (iso.length <= 10 ? 'T00:00:00' : 'Z'))
  if (isNaN(d)) return iso
  return d.toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  })
}

// A stored timestamp as date plus time, for "last updated" lines.
function formatWhen(iso) {
  if (!iso) return ''
  const d = new Date(String(iso).replace(' ', 'T') + 'Z')
  if (isNaN(d)) return iso
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit',
  })
}

// The date a recording belongs to: when the audio was recorded
// (recorded_at, backfilled server-side), not when it was processed.
function recordingDate(r) {
  return (r.recorded_at || r.created_at || '').slice(0, 10)
}

// Deterministic pastel per speaker so chips stay stable across renders.
function speakerHue(label) {
  let h = 0
  for (const c of String(label)) h = (h * 31 + c.charCodeAt(0)) % 360
  return h
}

// Chip colors are OKLCH: only the hue is per-person (0-360); the
// theme-aware lightness/chroma live in these tokens, mirrored from
// styles.css (change both together). OKLCH lightness is perceptually
// even across hues, which is what keeps the FULL wheel readable.
export const CHIP_TOKENS = {
  light: { bg: [0.955, 0.055], text: [0.43, 0.055] },
  dark: { bg: [0.31, 0.07], text: [0.86, 0.07] },
}

// The ten quick-pick presets, as OKLCH hues. Plain numbers: exactly
// what gets stored, so a preset and a wheel pick are the same thing.
const PRESET_HUES = {
  red: 25, orange: 55, gold: 95, green: 148, teal: 178, cyan: 215,
  blue: 260, violet: 300, plum: 330, pink: 355,
}

// OKLCH -> linear sRGB (clipped to gamut). WCAG relative luminance is
// defined on linear sRGB, so contrast falls straight out.
function oklchLuminance(L, C, H) {
  const hr = (H * Math.PI) / 180
  const a = C * Math.cos(hr)
  const b = C * Math.sin(hr)
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
  const bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  const clip = (v) => Math.min(1, Math.max(0, v))
  return 0.2126 * clip(r) + 0.7152 * clip(g) + 0.0722 * clip(bl)
}

// WCAG contrast between a chip's text and background at this hue.
export function contrastForHue(hue, theme) {
  const t = CHIP_TOKENS[theme]
  const yBg = oklchLuminance(t.bg[0], t.bg[1], hue)
  const yText = oklchLuminance(t.text[0], t.text[1], hue)
  const [hi, lo] = yBg > yText ? [yBg, yText] : [yText, yBg]
  return (hi + 0.05) / (lo + 0.05)
}

// Normalize a candidate hue and, should any hue ever dip under 4.5:1
// in either theme, snap to the nearest one that clears it. With the
// current tokens every hue passes, so this is a wraparound plus a
// guarantee, not a visible behavior.
export function clampHue(hue) {
  const norm = ((Math.round(Number(hue) || 0) % 360) + 360) % 360
  const ok = (h) =>
    contrastForHue(h, 'light') >= 4.5 && contrastForHue(h, 'dark') >= 4.5
  if (ok(norm)) return norm
  for (let d = 1; d <= 180; d++) {
    if (ok((norm + d) % 360)) return (norm + d) % 360
    if (ok((norm - d + 360) % 360)) return (norm - d + 360) % 360
  }
  return norm
}

// Per-person color overrides, loaded once and shared app-wide so a
// change repaints every chip everywhere.
const SpeakerColorsCtx = React.createContext({
  colors: {}, setColor: async () => {},
})

// The style for a person's chip: their chosen hue number, or the
// automatic per-name hue when unset ("Auto"). An override previews a
// candidate (number, or 'auto') without committing.
function chipStyle(colors, name, override) {
  const chosen = override !== undefined ? override : colors[name]
  const hue = chosen == null || chosen === 'auto'
    ? speakerHue(name || 'unknown') : Number(chosen)
  return { '--hue': hue }
}

function useChipStyle() {
  const { colors } = React.useContext(SpeakerColorsCtx)
  return (name) => chipStyle(colors, name)
}

function deleteWarning(what) {
  return `Delete ${what}?\n\n` +
    'This permanently removes: the audio file from inbox/, the ' +
    'transcript and its segments, keyword search entries, semantic ' +
    'embeddings, the calendar link, commitments from this recording, ' +
    'your notes and their enhanced version, ' +
    'and its contribution to any cached person dossier. The filename ' +
    'stays in the sync ledger, so a deleted memo never re-syncs from ' +
    'the iPhone.\n\n' +
    'Enrolled voices survive, and speaker names in other recordings ' +
    'are untouched.\n\n' +
    'Deletion is immediate and permanent. It cannot be undone.'
}

// Navigate via a temporary link so the zip arrives as a download.
function downloadExport(ids) {
  const a = document.createElement('a')
  a.href = `/api/export?ids=${ids.join(',')}`
  document.body.appendChild(a)
  a.click()
  a.remove()
}

// Does this query read like a natural-language question rather than
// keywords? Question mark, an interrogative opener, or plain length.
// Decides whether a search also fires the (LLM-backed) Ask in parallel.
function looksLikeQuestion(q) {
  const t = q.trim().toLowerCase()
  if (!t) return false
  if (t.endsWith('?')) return true
  if (/^(what|who|when|where|why|how|did|does|is|are|should)\b/.test(t)) {
    return true
  }
  return t.split(/\s+/).length > 6
}

// Defaults and hard limits for the draggable pane seams. The detail
// pane keeps its minimum implicitly: with both other panes at max on a
// 1300px window it still has ~350px.
const PANE_DEFAULTS = { sidebar: 240, list: 340 }
const PANE_LIMITS = { sidebar: [140, 320], list: [260, 560] }

// A draggable seam between panes. Pointer drags resize live, arrow
// keys nudge, double-click resets the seam to its default width.
function PaneHandle({ label, value, min, max, onChange, onReset }) {
  const start = useRef(null)
  return (
    <div
      className="pane-handle"
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      tabIndex={0}
      title={`Drag to resize. Double-click to reset.`}
      onPointerDown={(e) => {
        e.preventDefault()
        e.currentTarget.setPointerCapture?.(e.pointerId)
        start.current = { x: e.clientX, v: value }
      }}
      onPointerMove={(e) => {
        if (start.current == null) return
        onChange(start.current.v + e.clientX - start.current.x)
      }}
      onPointerUp={() => { start.current = null }}
      onPointerCancel={() => { start.current = null }}
      onDoubleClick={onReset}
      onKeyDown={(e) => {
        if (e.key === 'ArrowLeft') { e.preventDefault(); onChange(value - 16) }
        if (e.key === 'ArrowRight') { e.preventDefault(); onChange(value + 16) }
      }}
    />
  )
}

// A recording that came in from Voice Memos keeps its memo filename;
// in-app captures carry the -meeting/-inperson marker instead. Used to
// decide whether "Re-sync from source" can exist for a row.
function isVoiceMemoName(name) {
  return /^\d{8} \d{6}/.test(name || '')
    && !name.includes('-meeting') && !name.includes('-inperson')
}

function RowMenu({ onDelete, onExport, onRename, onMove,
                   onRetry, onResync, onReprocess }) {
  const [open, setOpen] = useState(false)
  const [sub, setSub] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return
    const close = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const pick = (action) => (e) => {
    e.stopPropagation()
    setOpen(false)
    setSub(false)
    action()
  }
  return (
    <span className="row-menu" ref={ref}>
      <button
        className="menu-trigger"
        aria-label="More actions"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => { e.stopPropagation(); setOpen(!open); setSub(false) }}
      >
        ⋯
      </button>
      {open && (
        <span className="menu" role="menu">
          {onRename && (
            <button role="menuitem" onClick={pick(onRename)}>Rename</button>
          )}
          {onMove && (
            <button role="menuitem" onClick={pick(onMove)}>
              Move to folder…
            </button>
          )}
          {onRetry && (
            <button role="menuitem" onClick={pick(onRetry)}>Retry</button>
          )}
          {onReprocess && (
            <button role="menuitem" aria-haspopup="menu" aria-expanded={sub}
              onClick={(e) => { e.stopPropagation(); setSub(!sub) }}>
              Reprocess…
            </button>
          )}
          {onReprocess && sub && (
            <>
              <button role="menuitem" className="submenu-item"
                onClick={pick(() => onReprocess('everything'))}>
                Everything
              </button>
              <button role="menuitem" className="submenu-item"
                onClick={pick(() => onReprocess('speakers'))}>
                Speakers only
              </button>
              <button role="menuitem" className="submenu-item"
                onClick={pick(() => onReprocess('notes'))}>
                Notes only
              </button>
            </>
          )}
          {onResync && (
            <button role="menuitem" onClick={pick(onResync)}>
              Re-sync from source
            </button>
          )}
          <button role="menuitem" onClick={pick(onExport)}>Export</button>
          <button role="menuitem" className="danger" onClick={pick(onDelete)}>
            Delete
          </button>
        </span>
      )}
    </span>
  )
}

// Persistent folder sidebar: All, Unfiled, then every folder with live
// counts. One click filters; folder create/rename/delete live here.
function FolderSidebar({ folders, total, unfiled, current, onSelect,
                        onCreate, onRename, onDelete }) {
  const item = (key, label, count) => (
    <button
      className={`folder-row ${current === key ? 'active' : ''}`}
      onClick={() => onSelect(key)}>
      <span className="folder-name">{label}</span>
      <span className="folder-count">{count}</span>
    </button>
  )
  return (
    <aside className="folder-sidebar">
      <div className="section-head">
        <h3>Folders</h3>
        <button className="link" onClick={onCreate}
          title="Create a new folder">+ New</button>
      </div>
      {item('', 'All', total)}
      {item('unfiled', 'Unfiled', unfiled)}
      {folders.map((f) => (
        <div key={f.id} className="folder-item">
          <button
            className={`folder-row ${current === String(f.id) ? 'active' : ''}`}
            onClick={() => onSelect(String(f.id))}>
            <span className="folder-name">{f.name}</span>
            <span className="folder-count">{f.count}</span>
          </button>
          <span className="folder-tools">
            <button className="link" aria-label={`Rename ${f.name}`}
              onClick={() => onRename(f)}>Rename</button>
            <button className="link danger" aria-label={`Delete ${f.name}`}
              onClick={() => onDelete(f)}>Delete</button>
          </span>
        </div>
      ))}
    </aside>
  )
}

function LevelMeter({ label, value, warn }) {
  // RMS is small; a sqrt curve makes normal speech fill a useful range.
  const pct = Math.min(100, Math.round(Math.sqrt(value || 0) * 140))
  return (
    <span className={`meter ${warn ? 'warn' : ''}`}>
      <span className="meter-label">{label}</span>
      <span className="meter-track" aria-hidden="true">
        <span className="meter-fill" style={{ width: `${pct}%` }} />
      </span>
    </span>
  )
}

function Spinner({ label }) {
  return (
    <div className="spinner-row" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

function StatusChip({ status }) {
  if (status === 'done') return null
  const busy = status === 'transcribing' || status === 'pending'
  return (
    <span className={`status-chip ${busy ? 'busy' : 'failed'}`}>
      {busy && <span className="pulse-dot" aria-hidden="true" />}
      {status}
    </span>
  )
}

function EmptyState({ filtered }) {
  if (filtered) {
    return (
      <div className="empty">
        <h2>No recordings match</h2>
        <p>Try clearing a filter or widening the date range.</p>
      </div>
    )
  }
  return (
    <div className="empty">
      <svg width="44" height="44" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
      </svg>
      <h2>No recordings yet</h2>
      <p>
        Tap Sync from iPhone, or drop an audio file into the
        <code> inbox/</code> folder. It will show up here with a
        transcript, speakers, and AI notes.
      </p>
    </div>
  )
}

function Summary({ summary }) {
  if (!summary) return null
  return (
    <section className="summary">
      <h3>AI Notes</h3>
      <p>{summary.summary}</p>
      {summary.decisions.length > 0 && (
        <>
          <h4>Decisions</h4>
          <ul>{summary.decisions.map((d, i) => <li key={i}>{d}</li>)}</ul>
        </>
      )}
      {summary.action_items.length > 0 && (
        <>
          <h4>Action items</h4>
          <ul className="actions">
            {summary.action_items.map((a, i) => (
              <li key={i}>
                <strong>{a.owner}</strong>: {a.task}
                {a.due_date ? <span className="due"> due {a.due_date}</span> : null}
              </li>
            ))}
          </ul>
        </>
      )}
      {summary.topics.length > 0 && (
        <p className="topics">
          {summary.topics.map((t, i) => (
            <span key={i} className="topic-chip">{t}</span>
          ))}
        </p>
      )}
    </section>
  )
}

function Detail({ recordingId, onBack, onDeleted }) {
  const hueFor = useChipStyle()
  const [detail, setDetail] = useState(null)
  const [working, setWorking] = useState(false)
  const [activeSegment, setActiveSegment] = useState(-1)
  const [translation, setTranslation] = useState(null)
  const [showTranslation, setShowTranslation] = useState(false)
  const [translating, setTranslating] = useState(false)
  const [translateError, setTranslateError] = useState(null)
  const [myNotes, setMyNotes] = useState('')
  const [notesStatus, setNotesStatus] = useState(null)
  const [enhanced, setEnhanced] = useState(null)
  const [enhancing, setEnhancing] = useState(false)
  const [enhanceError, setEnhanceError] = useState(null)
  const [undo, setUndo] = useState(null)
  // Every speaker edit reports its outcome here, so a change that did
  // nothing says so instead of looking like a dead button.
  const [speakerMsg, setSpeakerMsg] = useState(null)
  const [openLineMenu, setOpenLineMenu] = useState(null)
  const [lineSelectMode, setLineSelectMode] = useState(false)
  const [selectedLines, setSelectedLines] = useState(new Set())
  const audioRef = useRef(null)
  const notesTimer = useRef(null)

  const load = () =>
    fetch(`/api/recordings/${recordingId}`)
      .then((r) => r.json())
      .then(setDetail)

  useEffect(() => {
    setTranslation(null)
    setShowTranslation(false)
    setTranslateError(null)
    setNotesStatus(null)
    setEnhanceError(null)
    fetch(`/api/recordings/${recordingId}`)
      .then((r) => r.json())
      .then((d) => {
        setDetail(d)
        setMyNotes(d.notes?.raw_text || '')
        setEnhanced(d.notes?.enhanced || null)
      })
    return () => clearTimeout(notesTimer.current)
  }, [recordingId])

  const saveNotes = (text) =>
    fetch(`/api/recordings/${recordingId}/notes`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })

  const onNotesChange = (e) => {
    const text = e.target.value
    setMyNotes(text)
    setNotesStatus('Saving…')
    clearTimeout(notesTimer.current)
    notesTimer.current = setTimeout(async () => {
      await saveNotes(text)
      setNotesStatus('Saved')
    }, 800)
  }

  const doEnhance = async () => {
    setEnhancing(true)
    setEnhanceError(null)
    try {
      // Flush any pending autosave so the LLM sees the latest text.
      clearTimeout(notesTimer.current)
      await saveNotes(myNotes)
      setNotesStatus('Saved')
      const res = await fetch(
        `/api/recordings/${recordingId}/notes/enhance`, { method: 'POST' })
      const body = await res.json()
      if (!res.ok) {
        setEnhanceError(body.detail || 'Enhance failed')
      } else {
        setEnhanced(body.enhanced)
      }
    } catch {
      setEnhanceError('Enhance failed')
    } finally {
      setEnhancing(false)
    }
  }

  const targetLang = detail?.language === 'es' ? 'en' : 'es'
  const targetName = targetLang === 'es' ? 'Spanish' : 'English'

  const doTranslate = async () => {
    if (translation) { setShowTranslation(!showTranslation); return }
    setTranslating(true)
    setTranslateError(null)
    try {
      const res = await fetch(`/api/recordings/${recordingId}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target: targetLang }),
      })
      const body = await res.json()
      if (!res.ok) {
        setTranslateError(body.detail || 'Translation failed')
      } else {
        setTranslation(body)
        setShowTranslation(true)
      }
    } catch {
      setTranslateError('Translation failed')
    } finally {
      setTranslating(false)
    }
  }

  const togglePrivileged = async () => {
    await fetch(`/api/recordings/${recordingId}/privileged`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ privileged: !detail.privileged }),
    })
    await load()
  }

  // Answer "Which meeting was this?". null means neither, which
  // dismisses the recording rather than matching it.
  const chooseEvent = async (eventId) => {
    await fetch(`/api/recordings/${recordingId}/calendar/choose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event_id: eventId }),
    })
    await load()
  }

  const generate = async () => {
    setWorking(true)
    try {
      await fetch(`/api/recordings/${recordingId}/summarize`, { method: 'POST' })
      await load()
    } finally {
      setWorking(false)
    }
  }

  const tid = detail?.transcript_id

  // Undo: snapshot the affected lines, run the action, then offer a
  // toast that restores that exact snapshot. Single-step is enough.
  const runWithUndo = async (segmentIds, run, message) => {
    let snapshot = []
    if (segmentIds.length) {
      const res = await fetch(
        `/api/transcripts/${tid}/segments/states?ids=${segmentIds.join(',')}`)
      if (!res.ok) {
        setSpeakerMsg('Could not read the current lines, so nothing changed.')
        return
      }
      snapshot = await res.json()
    }
    const res = await run()
    if (res && !res.ok) {
      let detail = null
      try { detail = (await res.json()).detail } catch { /* no body */ }
      setSpeakerMsg(detail || `That change failed (${res.status}).`)
      return
    }
    await load()
    setSpeakerMsg(message)
    setUndo({ message, snapshot })
  }

  const doUndo = async () => {
    if (!undo) return
    let res
    try {
      res = await fetch(`/api/transcripts/${tid}/segments/restore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ segments: undo.snapshot }),
      })
    } catch {
      setSpeakerMsg('Could not reach the app, so the undo did not happen.')
      return
    }
    if (!res.ok) {
      let detail = null
      try { detail = (await res.json()).detail } catch { /* no body */ }
      // Keep the toast: the undo is still available to retry.
      setSpeakerMsg(detail || `Undo failed (${res.status}).`)
      return
    }
    setUndo(null)
    setSpeakerMsg('Undone.')
    await load()
  }

  // Line-scoped: assign one line to a speaker (from its menu).
  const assignLine = (seg, name) =>
    runWithUndo([seg.id], () =>
      fetch(`/api/transcripts/${tid}/segments/${seg.id}/speaker`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      }), `Line assigned to ${name}`)

  const newSpeakerForLine = (seg) => {
    const name = window.prompt('New speaker name for this line:', '')
    if (name === null) return
    if (!name.trim()) {
      setSpeakerMsg('That name was empty, so nothing changed.')
      return
    }
    assignLine(seg, name.trim())
  }

  const confirmLine = (seg) =>
    runWithUndo([seg.id], () =>
      fetch(`/api/transcripts/${tid}/segments/${seg.id}/confirm`,
        { method: 'POST' }), `Confirmed ${seg.speaker} on this line`)

  const revertLine = (seg) =>
    runWithUndo([seg.id], () =>
      fetch(`/api/transcripts/${tid}/segments/${seg.id}/revert`,
        { method: 'POST' }), 'Line reverted')

  const assignSelectedLines = async () => {
    const ids = [...selectedLines]
    if (!ids.length) return
    const name = window.prompt(
      `Assign the ${ids.length} selected line${ids.length === 1 ? '' : 's'}`
      + ' to:', '')
    if (name === null) return
    if (!name.trim()) {
      setSpeakerMsg('That name was empty, so nothing changed.')
      return
    }
    await runWithUndo(ids, () =>
      fetch(`/api/transcripts/${tid}/segments/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids, name: name.trim() }),
      }), `${ids.length} lines assigned to ${name.trim()}`)
    setSelectedLines(new Set())
    setLineSelectMode(false)
  }

  // Group-scoped (legend / Voices tab only): blast radius spelled out.
  const renameEverywhere = async (speaker) => {
    const newLabel = window.prompt(
      `Rename "${speaker}" everywhere in this recording to:`, speaker)
    if (newLabel === null) return
    if (!newLabel.trim()) {
      setSpeakerMsg('That name was empty, so nothing changed.')
      return
    }
    if (newLabel === speaker) {
      setSpeakerMsg(`Everyone here is already called "${speaker}".`)
      return
    }
    const readError = async (res) => {
      try { return (await res.json()).detail } catch { return null }
    }
    // One attempt, including the merge guard's confirmation. Returns
    // the parsed body, or null when it failed or the user backed out
    // (a message is already on screen in that case).
    let forced = false
    const attempt = async (opts) => {
      let res
      try {
        res = await fetch(`/api/transcripts/${tid}/speakers`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            old_label: speaker, new_label: newLabel,
            force: forced, include_pinned: false, ...opts,
          }),
        })
      } catch {
        setSpeakerMsg('Could not reach the app, so nothing changed.')
        return null
      }
      // Merge guard: renaming onto an existing name needs a yes.
      if (res.status === 409) {
        const detail = await readError(res)
        if (!window.confirm(`${detail}\n\nMerge them anyway?`)) {
          setSpeakerMsg('Merge cancelled, nothing changed.')
          return null
        }
        forced = true
        return attempt({ ...opts, force: true })
      }
      if (!res.ok) {
        const detail = await readError(res)
        setSpeakerMsg(detail || `Rename failed (${res.status}).`)
        return null
      }
      try { return await res.json() } catch { return {} }
    }

    let body = await attempt({})
    if (body === null) return

    // The silent dead end: every line of this speaker is pinned, so a
    // normal rename moves nothing. Say so, and offer the way through.
    if (body.changed === 0 && body.pinned_skipped > 0) {
      const n = body.pinned_skipped
      const ok = window.confirm(
        `Nothing changed: all ${n} of "${speaker}"'s line`
        + `${n === 1 ? ' is' : 's are'} pinned, because you set `
        + `${n === 1 ? 'it' : 'them'} by hand line by line.\n\n`
        + `Rename ${n === 1 ? 'that pinned line' : `those ${n} pinned lines`} `
        + `to "${newLabel}" as well?`)
      if (!ok) {
        setSpeakerMsg(
          `Nothing changed: ${n} pinned line${n === 1 ? '' : 's'} left as `
          + `"${speaker}".`)
        return
      }
      body = await attempt({ include_pinned: true })
      if (body === null) return
    }

    if (body.changed === 0) {
      setSpeakerMsg(`Nothing changed: no lines are labelled "${speaker}".`)
    } else {
      setSpeakerMsg(
        `Renamed ${body.changed} line${body.changed === 1 ? '' : 's'} to `
        + `"${newLabel}".`
        + (body.enrolled_samples
          ? ' Voice enrolled; naming it across the archive.' : ''))
    }
    await load()
  }

  const revertAllForSpeaker = async (speaker, autoCount) => {
    if (!window.confirm(
      `Revert all ${autoCount} auto-tagged line${autoCount === 1 ? '' : 's'} `
      + `for "${speaker}"?\n\nThey go back to their SPEAKER_XX labels, and `
      + 'this voice will not be auto-applied to them again. Lines you set '
      + 'by hand are not affected.')) return
    const res = await fetch(`/api/transcripts/${tid}/speakers/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: speaker }),
    })
    if (!res.ok) {
      let detail = null
      try { detail = (await res.json()).detail } catch { /* no body */ }
      setSpeakerMsg(detail || `Revert failed (${res.status}).`)
      return
    }
    const body = await res.json().catch(() => ({}))
    setSpeakerMsg(
      `Reverted ${body.reverted ?? autoCount} auto-tagged line`
      + `${(body.reverted ?? autoCount) === 1 ? '' : 's'} for "${speaker}".`)
    await load()
  }

  const dismissMergedName = async () => {
    try {
      await fetch(`/api/recordings/${recordingId}/merged-name/dismiss`,
        { method: 'POST' })
    } catch { /* the note is only a suggestion */ }
    await load()
  }

  const dismissSuspect = async () => {
    try {
      await fetch(
        `/api/recordings/${recordingId}/diarization-suspect/dismiss`,
        { method: 'POST' })
    } catch { /* the banner is only a suggestion */ }
    await load()
  }

  const resetSpeakers = async () => {
    if (!window.confirm(
      'Reset speakers for this recording?\n\n' +
      'Every speaker name in this recording (including ones you set) ' +
      'goes back to fresh SPEAKER_XX labels from a new diarization ' +
      'pass, and enrolled voices are then re-applied automatically ' +
      'where they match. Other recordings are not affected.\n\n' +
      'This takes a few minutes.')) return
    const res = await fetch(`/api/recordings/${recordingId}/speakers/reset`,
      { method: 'POST' })
    if (!res.ok) {
      let detail = null
      try { detail = (await res.json()).detail } catch { /* no body */ }
      setSpeakerMsg(detail || `Speaker reset failed (${res.status}).`)
      return
    }
    setSpeakerMsg('Resetting speakers; this takes a few minutes.')
    await load()
  }

  // Poll while a speaker reset runs in the background.
  useEffect(() => {
    if (detail?.reset_status === 'running') {
      const t = setTimeout(load, 4000)
      return () => clearTimeout(t)
    }
  }, [detail])

  // The undo toast fades on its own after a while.
  useEffect(() => {
    if (!undo) return
    const t = setTimeout(() => setUndo(null), 8000)
    return () => clearTimeout(t)
  }, [undo])

  // Reset line-selection when switching recordings.
  useEffect(() => {
    setLineSelectMode(false)
    setSelectedLines(new Set())
    setUndo(null)
    setSpeakerMsg(null)
    setOpenLineMenu(null)
  }, [recordingId])

  const seekTo = (seconds) => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = seconds
    audio.play()
  }

  const onTimeUpdate = () => {
    const audio = audioRef.current
    if (!audio || !detail) return
    const t = audio.currentTime
    setActiveSegment(detail.segments.findIndex(
      (s) => t >= s.start && t < s.end,
    ))
  }

  const deleteThis = async () => {
    if (!window.confirm(deleteWarning(
      `"${detail.summary?.title || detail.file_name}"`))) return
    const res = await fetch(`/api/recordings/${recordingId}`,
      { method: 'DELETE' })
    if (res.ok) onDeleted()
  }

  const renameTitle = async () => {
    const current = detail.manual_title || detail.calendar_event?.title
      || detail.summary?.title || ''
    const t = window.prompt(
      'Title for this recording (leave empty to go back to the '
      + 'automatic title):', current)
    if (t === null) return
    await fetch(`/api/recordings/${recordingId}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: t }),
    })
    await load()
  }

  if (!detail) return <Spinner label="Loading recording…" />
  const hasSegments = detail.segments.length > 0
  const shownSummary = showTranslation && translation?.summary
    ? translation.summary : detail.summary
  return (
    <article>
      <button className="back" onClick={onBack}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
        </svg>
        All recordings
      </button>
      <div className="detail-head">
        <h2>
          {detail.manual_title || detail.calendar_event?.title
            || shownSummary?.title || detail.file_name}
          {detail.manual_title && (
            <span className="muted title-pin" title="You named this recording yourself; automatic titles will never overwrite it."> ✎</span>
          )}
        </h2>
        <RowMenu
          onRename={renameTitle}
          onDelete={deleteThis}
          onExport={() => downloadExport([recordingId])}
        />
      </div>
      <p className="meta">
        {detail.file_name} · {formatDuration(detail.duration_seconds)} · {formatDate(recordingDate(detail))}
      </p>
      {detail.calendar_event && (
        <div className="event-card">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5" />
          </svg>
          <span className="event-body">
            <strong>{detail.calendar_event.title}</strong>
            {detail.calendar_event.attendees.length > 0 && (
              <span className="muted">
                {' '}with {detail.calendar_event.attendees.join(', ')}
              </span>
            )}
          </span>
          <button className="link" title="Not this event? Unlink it."
            onClick={async () => {
              await fetch(`/api/recordings/${recordingId}/calendar/unlink`,
                { method: 'POST' })
              await load()
            }}>
            Unlink
          </button>
        </div>
      )}
      {(detail.calendar_candidates || []).length > 0 && (
        <div className="event-card event-ambiguous">
          <span className="event-body">
            <strong>Which meeting was this?</strong>
            <span className="muted">
              {' '}Two events fit this recording about equally well, so
              nothing was matched.
            </span>
            <span className="event-choices">
              {detail.calendar_candidates.map((c) => (
                <button key={c.id} className="ghost"
                  onClick={() => chooseEvent(c.id)}>
                  {c.title}
                  {c.attendees.length > 0 && (
                    <span className="muted">
                      {' '}· {c.attendees.join(', ')}
                    </span>
                  )}
                </button>
              ))}
              <button className="link" onClick={() => chooseEvent(null)}>
                Neither
              </button>
            </span>
          </span>
        </div>
      )}
      <p className="detail-actions">
        <button className="ghost" onClick={togglePrivileged}
          title="Privileged recordings are analyzed only by a local model. Nothing is sent to the cloud.">
          {detail.privileged ? '🔒 Privileged' : 'Mark privileged'}
        </button>
        {detail.transcript_id && (
          <button className="ghost" onClick={doTranslate} disabled={translating}>
            {translating ? 'Translating…'
              : translation
                ? (showTranslation ? 'Show original' : `Show ${targetName}`)
                : `Translate to ${targetName}`}
          </button>
        )}
        {translateError && <span className="error-msg">{translateError}</span>}
      </p>
      <audio
        ref={audioRef}
        controls
        preload="metadata"
        src={`/api/recordings/${recordingId}/audio`}
        onTimeUpdate={onTimeUpdate}
        onEnded={() => setActiveSegment(-1)}
        className="player"
        data-testid="audio-player"
      />
      {/* Below 1700px these wrappers are invisible block flow; in the
          ultrawide detail pane they become two columns: notes left,
          transcript right. */}
      <div className="detail-cols">
      <div className="detail-col">
      {shownSummary ? (
        <Summary summary={shownSummary} />
      ) : detail.transcript_id ? (
        <button className="primary" onClick={generate} disabled={working}>
          {working ? 'Generating…' : 'Generate AI notes'}
        </button>
      ) : null}
      <section className="my-notes">
        <div className="section-head">
          <h3>My notes</h3>
          {notesStatus && (
            <span className="muted" role="status">{notesStatus}</span>
          )}
          {detail.transcript_id && (
            <button className="ghost" onClick={doEnhance}
              disabled={enhancing || !myNotes.trim()}
              title="Merge your notes with the transcript into structured notes. Your own text is never changed.">
              {enhancing ? 'Enhancing…' : enhanced ? 'Re-enhance' : 'Enhance'}
            </button>
          )}
        </div>
        <textarea
          value={myNotes}
          onChange={onNotesChange}
          rows={4}
          aria-label="My notes"
          placeholder="Jot anything during or after the meeting. Saved as you type."
        />
        {enhanceError && <span className="error-msg">{enhanceError}</span>}
        {enhanced && (
          <div className="enhanced-notes summary">
            <h4>Enhanced notes</h4>
            {enhanced.sections.map((s, i) => (
              <React.Fragment key={i}>
                <h4 className="enhanced-heading">{s.heading}</h4>
                <ul>{s.bullets.map((b, j) => <li key={j}>{b}</li>)}</ul>
              </React.Fragment>
            ))}
          </div>
        )}
      </section>
      </div>
      <div className="detail-col">
      <div className="section-head">
        <h3>Transcript{showTranslation ? ` (${targetName})` : ''}</h3>
        {detail.reset_status === 'running' ? (
          <Spinner label="Resetting speakers…" />
        ) : (
          <>
            {detail.reset_status === 'failed' && (
              <span className="error-msg">Speaker reset failed.</span>
            )}
            {hasSegments && !showTranslation && (
              <button className="link"
                onClick={() => {
                  setSelectedLines(new Set())
                  setLineSelectMode(!lineSelectMode)
                }}>
                {lineSelectMode ? 'Cancel' : 'Select lines'}
              </button>
            )}
          </>
        )}
      </div>
      {detail.merged_name_suspect && !speakerMsg && (
        <p className="speaker-msg suspect-msg" role="status">
          Two distinct voices appear under one name
          {' ('}<strong>{detail.merged_name_suspect}</strong>{'), '}
          so part of this transcript is probably credited to the wrong
          person. Nothing has been changed.
          <button className="link" onClick={resetSpeakers}>
            Reset speakers
          </button>
          <button className="link" onClick={() => {
            setSelectedLines(new Set())
            setLineSelectMode(true)
          }}>
            Split by hand
          </button>
          <button className="link" onClick={dismissMergedName}>Dismiss</button>
        </p>
      )}
      {detail.diarization_suspect && !detail.merged_name_suspect
        && !speakerMsg && (
        <p className="speaker-msg suspect-msg" role="status">
          This recording is {Math.round((detail.duration_seconds || 0) / 60)}
          {' '}minutes long but came out as a single speaker, which usually
          means speaker detection merged people together. Reprocessing with
          the number of speakers often fixes it.
          <button className="link" onClick={dismissSuspect}>Dismiss</button>
        </p>
      )}
      {speakerMsg && (
        <p className="speaker-msg" role="status">
          {speakerMsg}
          <button className="link" onClick={() => setSpeakerMsg(null)}>
            Dismiss
          </button>
        </p>
      )}
      {!showTranslation && hasSegments && (
        <SpeakerLegend
          segments={detail.segments}
          onRename={renameEverywhere}
          onRevertAll={revertAllForSpeaker}
          onReset={resetSpeakers}
          resetting={detail.reset_status === 'running'}
        />
      )}
      {lineSelectMode && (
        <div className="bulk-bar">
          <span className="muted">{selectedLines.size} line
            {selectedLines.size === 1 ? '' : 's'} selected</span>
          <span className="bulk-actions">
            <button className="ghost" disabled={selectedLines.size === 0}
              onClick={assignSelectedLines}>
              Assign selected to…
            </button>
          </span>
        </div>
      )}
      {showTranslation && translation ? (
        <p className="transcript">{translation.full_text}</p>
      ) : hasSegments ? (
        <div className="segments">
          {detail.segments.map((s, i) => (
            <p key={i}
              className={`segment ${i === activeSegment ? 'active' : ''}`}>
              {lineSelectMode && (
                <input type="checkbox" className="select-box line-box"
                  checked={selectedLines.has(s.id)}
                  aria-label={`Select line ${i + 1}`}
                  onChange={() => setSelectedLines((prev) => {
                    const next = new Set(prev)
                    if (next.has(s.id)) next.delete(s.id)
                    else next.add(s.id)
                    return next
                  })} />
              )}
              <span className="speaker-cell">
                <span className="line-speaker-wrap">
                  <button
                    className={`speaker ${s.auto ? 'auto' : ''} ${s.pinned ? 'pinned' : ''}`}
                    style={hueFor(s.speaker)}
                    title="Set who said this line"
                    aria-label={`Line ${i + 1} speaker ${s.speaker || 'unknown'}`}
                    onClick={() => setOpenLineMenu(
                      openLineMenu === s.id ? null : s.id)}
                    disabled={!s.speaker}
                  >
                    {s.speaker || 'unknown'}
                  </button>
                  {openLineMenu === s.id && (
                    <LineSpeakerMenu
                      seg={s}
                      speakers={detail.segments
                        .map((x) => x.speaker).filter(Boolean)}
                      onPick={(name) => { setOpenLineMenu(null); assignLine(s, name) }}
                      onNew={() => { setOpenLineMenu(null); newSpeakerForLine(s) }}
                      onClose={() => setOpenLineMenu(null)}
                    />
                  )}
                </span>
                {s.auto && (
                  <>
                    <button className="tag-action confirm"
                      title="This line is right: confirm just this line"
                      aria-label={`Confirm line ${i + 1}`}
                      onClick={() => confirmLine(s)}>
                      ✓
                    </button>
                    <button className="tag-action reject"
                      title="Wrong on this line: revert just this line"
                      aria-label={`Revert line ${i + 1}`}
                      onClick={() => revertLine(s)}>
                      ✕
                    </button>
                  </>
                )}
              </span>
              <button
                className="segment-text"
                title="Play from here"
                onClick={() => seekTo(s.start)}
              >
                {s.text}
              </button>
            </p>
          ))}
        </div>
      ) : (
        <p className="transcript">{detail.full_text || 'No transcript yet.'}</p>
      )}
      </div>
      </div>
      {undo && (
        <div className="undo-toast" role="status">
          <span>{undo.message}</span>
          <button className="link" onClick={doUndo}>Undo</button>
        </div>
      )}
    </article>
  )
}

// The speaker legend: every speaker in this recording with its color
// and line count. Group actions (blast radius spelled out) live here,
// not on individual lines.
function SpeakerLegend({ segments, onRename, onRevertAll, onReset,
                        resetting }) {
  const hueFor = useChipStyle()
  const speakers = useMemo(() => {
    const by = new Map()
    for (const s of segments) {
      if (!s.speaker) continue
      const e = by.get(s.speaker) || { name: s.speaker, count: 0, auto: 0 }
      e.count += 1
      if (s.auto) e.auto += 1
      by.set(s.speaker, e)
    }
    return [...by.values()].sort((a, b) => b.count - a.count)
  }, [segments])

  return (
    <div className="legend">
      {speakers.map((sp) => (
        <div key={sp.name} className="legend-item">
          <span className="speaker" style={hueFor(sp.name)}>
            {sp.name}
          </span>
          <span className="muted">
            {sp.count} line{sp.count === 1 ? '' : 's'}
            {sp.auto > 0 ? ` · ${sp.auto} auto` : ''}
          </span>
          <button className="link" onClick={() => onRename(sp.name)}>
            Rename everywhere
          </button>
          {sp.auto > 0 && (
            <button className="link danger"
              onClick={() => onRevertAll(sp.name, sp.auto)}>
              Revert all {sp.auto} auto-tagged
            </button>
          )}
        </div>
      ))}
      <button className="link danger reset-all" onClick={onReset}
        disabled={resetting}
        title="Start over: fresh SPEAKER_XX labels from a new diarization pass, then enrolled voices re-apply automatically.">
        Reset speakers
      </button>
    </div>
  )
}

// The line-scoped speaker picker: "This line is: [each speaker] / new".
function LineSpeakerMenu({ seg, speakers, onPick, onNew, onClose }) {
  const ref = useRef(null)
  useEffect(() => {
    const close = (e) => {
      if (ref.current && !ref.current.contains(e.target)) onClose()
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])
  const unique = [...new Set(speakers)]
  return (
    <span className="menu line-menu" role="menu" ref={ref}>
      <span className="menu-label">This line is:</span>
      {unique.map((name) => (
        <button key={name} role="menuitem"
          className={name === seg.speaker ? 'current' : ''}
          onClick={() => onPick(name)}>
          {name}
        </button>
      ))}
      <button role="menuitem" onClick={onNew}>someone new…</button>
    </span>
  )
}

function Person({ name, onBack, onOpenRecording }) {
  const hueFor = useChipStyle()
  const [data, setData] = useState(null)
  const [building, setBuilding] = useState(false)

  const load = () =>
    fetch(`/api/people/${encodeURIComponent(name)}`)
      .then((r) => r.json()).then(setData)

  useEffect(() => { load() }, [name])

  const buildDossier = async () => {
    setBuilding(true)
    try {
      await fetch(`/api/people/${encodeURIComponent(name)}/dossier`,
        { method: 'POST' })
      await load()
    } finally {
      setBuilding(false)
    }
  }

  if (!data) return <Spinner label="Loading person…" />
  return (
    <article>
      <button className="back" onClick={onBack}>&larr; People</button>
      <h2>
        <span className="speaker" style={hueFor(name)}>
          {name}
        </span>
      </h2>
      {data.dossier ? (
        <section className="summary">
          <div className="section-head">
            <h3>Dossier</h3>
            <button className="link" onClick={buildDossier} disabled={building}>
              {building ? 'Updating…' : 'Refresh'}
            </button>
          </div>
          <p>{data.dossier.summary}</p>
          {data.dossier.cares_about.length > 0 && (
            <>
              <h4>Cares about</h4>
              <ul>{data.dossier.cares_about.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </>
          )}
          {data.dossier.commitments_made.length > 0 && (
            <>
              <h4>Commitments made</h4>
              <ul>{data.dossier.commitments_made.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </>
          )}
          {data.dossier.follow_ups_owed.length > 0 && (
            <>
              <h4>Follow-ups owed</h4>
              <ul>{data.dossier.follow_ups_owed.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </>
          )}
        </section>
      ) : (
        <button className="primary" onClick={buildDossier} disabled={building}>
          {building ? 'Building…' : 'Build dossier'}
        </button>
      )}

      {data.action_items.length > 0 && (
        <>
          <h3>Their action items</h3>
          <ul className="list">
            {data.action_items.map((a, i) => (
              <li key={i} className="plain-row">
                {a.task}
                {a.due_date ? <span className="due"> due {a.due_date}</span> : null}
                <span className="muted"> — {a.title || 'Untitled'}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <h3>Meetings ({data.meetings.length})</h3>
      <ul className="list">
        {data.meetings.map((m) => (
          <li key={m.recording_id}>
            <button className="row" onClick={() => onOpenRecording(m.recording_id)}>
              <span className="row-title">{m.title || 'Untitled'}</span>
              <span className="row-meta">{formatDate(m.date || m.created_at)}</span>
            </button>
          </li>
        ))}
      </ul>
    </article>
  )
}

// The color picker on a person's row: a swatch that opens a palette of
// the ten presets plus Auto. Hovering previews live (in the palette AND
// on the person's row chip); clicking commits and persists.
function PersonColorSwatch({ name, onPreview }) {
  const { colors, setColor } = React.useContext(SpeakerColorsCtx)
  const [open, setOpen] = useState(false)
  const [hover, setHover] = useState(null)
  const ref = useRef(null)
  useEffect(() => {
    if (!open) return
    const close = (e) => {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false); setHover(null); onPreview(null)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const current = colors[name] != null ? Number(colors[name]) : 'auto'
  // The wheel's working value: last stored hue, else the auto hue.
  const [draft, setDraft] = useState(() =>
    current === 'auto' ? speakerHue(name) : current)
  const shown = hover != null ? hover : current
  const shownHue = shown === 'auto' ? null : Number(shown)
  const enter = (v) => { setHover(v); onPreview(v) }
  const leave = () => { setHover(null); onPreview(null) }
  const commit = (v, close = false) => {
    setColor(name, v === 'auto' ? null : clampHue(v))
    if (close) { setOpen(false); setHover(null); onPreview(null) }
  }
  const alsoUsedBy = shownHue != null
    ? Object.entries(colors)
      .filter(([n, c]) => n !== name && Number(c) === shownHue)
      .map(([n]) => n)
    : []

  return (
    <span className="row-menu color-swatch-wrap" ref={ref}>
      <button className="color-swatch"
        aria-label={`Chip color for ${name}`}
        aria-haspopup="menu" aria-expanded={open}
        title="Chip color"
        style={chipStyle(colors, name)}
        onClick={(e) => { e.stopPropagation(); setOpen(!open) }} />
      {open && (
        <span className="menu color-pop" role="menu">
          <span className="color-preview">
            <span className="speaker" style={chipStyle(colors, name, shown)}>
              {name}
            </span>
            <span className="muted hue-readout">
              {shownHue != null ? `hue ${shownHue}` : 'auto'}
            </span>
          </span>
          <span className="color-grid">
            {Object.entries(PRESET_HUES).map(([key, hue]) => (
              <button key={key} role="menuitem"
                className={`color-cell ${current === hue ? 'current' : ''}`}
                aria-label={`Color ${key}`}
                style={{ '--hue': hue }}
                onMouseEnter={() => enter(hue)}
                onMouseLeave={leave}
                onFocus={() => enter(hue)}
                onBlur={leave}
                onClick={() => { setDraft(hue); commit(hue, true) }} />
            ))}
          </span>
          <span className="hue-row">
            <input type="range" className="hue-slider"
              min="0" max="359" step="1"
              aria-label={`Hue for ${name}`}
              value={draft}
              style={{ '--hue': draft }}
              onChange={(e) => {
                const v = clampHue(e.target.value)
                setDraft(v); enter(v)
              }}
              onPointerUp={() => commit(draft)}
              onKeyUp={(e) => {
                if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
                  'Home', 'End', 'PageUp', 'PageDown'].includes(e.key)) {
                  commit(draft)
                }
              }}
            />
            <input type="number" className="hue-num"
              min="0" max="359"
              aria-label={`Hue value for ${name}`}
              value={draft}
              onChange={(e) => {
                const v = clampHue(e.target.value)
                setDraft(v); enter(v)
              }}
              onBlur={() => leave()}
              onKeyDown={(e) => { if (e.key === 'Enter') commit(draft) }}
            />
          </span>
          <button role="menuitem" className="link color-auto"
            onMouseEnter={() => enter('auto')}
            onMouseLeave={leave}
            onClick={() => commit('auto', true)}>
            Auto{current === 'auto' ? ' ✓' : ''}
          </button>
          {alsoUsedBy.length > 0 && (
            <span className="muted color-note">
              Also used by {alsoUsedBy.join(', ')}. Same-colored speakers
              in one transcript are hard to tell apart.
            </span>
          )}
        </span>
      )}
    </span>
  )
}

function PeopleList({ onSelect }) {
  const { colors } = React.useContext(SpeakerColorsCtx)
  const [people, setPeople] = useState(null)
  // {name, color} while a palette hover previews a choice, so the row
  // chip shows it live too.
  const [preview, setPreview] = useState(null)

  useEffect(() => {
    fetch('/api/people').then((r) => r.json()).then(setPeople)
  }, [])

  if (!people) return <Spinner label="Loading people…" />
  if (people.length === 0) {
    return (
      <div className="empty">
        <h2>No named speakers yet</h2>
        <p>
          Open a recording and click a SPEAKER chip to give someone their
          real name. They will show up here with a dossier of their
          meetings, commitments, and topics.
        </p>
      </div>
    )
  }
  return (
    <ul className="list">
      {people.map((p) => (
        <li key={p.name} className="person-row">
          <PersonColorSwatch name={p.name}
            onPreview={(key) => setPreview(key ? { name: p.name, key } : null)} />
          <button className="row" onClick={() => onSelect(p.name)}>
            <span className="row-title">
              <span className="speaker"
                style={chipStyle(colors, p.name,
                  preview?.name === p.name ? preview.key : undefined)}>
                {p.name}
              </span>
            </span>
            <span className="row-meta">
              {p.meetings} meeting{p.meetings === 1 ? '' : 's'} · last {formatDate(p.last_seen)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}

function Voices() {
  const hueFor = useChipStyle()
  const [items, setItems] = useState(null)
  const [msg, setMsg] = useState(null)

  const load = () =>
    fetch('/api/voices').then((r) => r.json()).then(setItems)

  useEffect(() => { load() }, [])

  const rename = async (voice) => {
    const newName = window.prompt(
      `Rename the voice "${voice.name}" to:`, voice.name)
    if (newName === null) return
    if (!newName.trim()) {
      setMsg('That name was empty, so nothing changed.')
      return
    }
    if (newName === voice.name) {
      setMsg(`That voice is already called "${voice.name}".`)
      return
    }
    let res
    try {
      res = await fetch(
        `/api/voices/${encodeURIComponent(voice.name)}/rename`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_name: newName }),
        })
    } catch {
      setMsg('Could not reach the app, so nothing changed.')
      return
    }
    // Read the body only after the status: an unhandled server error
    // returns plain text, and parsing it first would throw silently.
    const body = await res.json().catch(() => ({}))
    setMsg(res.ok
      ? `Renamed to ${newName} (${body.segments_changed} segments updated)`
      : body.detail || `Rename failed (${res.status}).`)
    await load()
  }

  const remove = async (voice) => {
    if (!window.confirm(
      `Delete the enrolled voice "${voice.name}"?\n\n` +
      'Speakers it auto-tagged go back to their SPEAKER labels. ' +
      'Names you set by hand are kept.')) return
    let res
    try {
      res = await fetch(
        `/api/voices/${encodeURIComponent(voice.name)}`,
        { method: 'DELETE' })
    } catch {
      setMsg('Could not reach the app, so nothing changed.')
      return
    }
    const body = await res.json().catch(() => ({}))
    setMsg(res.ok
      ? `Deleted ${voice.name} (${body.segments_reverted} segments reverted)`
      : body.detail || `Delete failed (${res.status}).`)
    await load()
  }

  if (!items) return <Spinner label="Loading voices…" />
  return (
    <section>
      <div className="section-head">
        <h2>Voices</h2>
        {msg && <span className="sync-msg" role="status">{msg}</span>}
      </div>
      {items.length === 0 ? (
        <div className="empty">
          <h2>No enrolled voices yet</h2>
          <p>
            Open a recording and rename a SPEAKER chip to someone&apos;s real
            name. Their voice is enrolled and recognized in every other
            recording, past and future.
          </p>
        </div>
      ) : (
        <ul className="list">
          {items.map((v) => (
            <li key={v.name} className="voice-row">
              <span className="row-title">
                <span className="speaker" style={hueFor(v.name)}>
                  {v.name}
                </span>
              </span>
              <span className="row-meta">
                enrolled {formatDate(v.created_at)} ·
                {' '}{v.recordings} recording{v.recordings === 1 ? '' : 's'} ·
                {' '}{v.segments} segment{v.segments === 1 ? '' : 's'}
                {v.auto_segments > 0 ? ` · ${v.auto_segments} auto-tagged` : ''}
              </span>
              <span className="voice-actions">
                <button className="link" onClick={() => rename(v)}>Rename</button>
                <button className="link danger" onClick={() => remove(v)}>Delete</button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

// Due dates were stored and printed but never compared to today, so an
// overdue promise read exactly like one due next year. The state comes
// from the server (which owns "today" and OTTER_USER_NAME) and drives
// the chip, the sort, and the filters together.
const DUE_LABELS = {
  overdue: 'Overdue',
  'due-soon': 'Due soon',
}

function dueChip(item) {
  if (item.status === 'done') return null
  const label = DUE_LABELS[item.due_state]
  if (!label) return null
  const d = item.days_until
  let detail = ''
  if (item.due_state === 'overdue' && d != null) {
    detail = d === -1 ? ' · 1 day' : ` · ${Math.abs(d)} days`
  } else if (item.due_state === 'due-soon' && d != null) {
    detail = d === 0 ? ' · today' : d === 1 ? ' · tomorrow' : ` · ${d} days`
  }
  return (
    <span className={`topic-chip due-${item.due_state}`}>
      {label}{detail}
    </span>
  )
}

function Commitments({ onOpenRecording }) {
  const [items, setItems] = useState(null)
  const [filter, setFilter] = useState('open')
  const [dueFilter, setDueFilter] = useState('')
  const [whose, setWhose] = useState('')

  const load = () => {
    const params = new URLSearchParams()
    if (dueFilter) params.set('due', dueFilter)
    if (whose === 'mine') params.set('mine', 'true')
    if (whose === 'theirs') params.set('mine', 'false')
    const qs = params.toString()
    return fetch(`/api/commitments${qs ? `?${qs}` : ''}`)
      .then((r) => r.json()).then(setItems)
  }

  useEffect(() => { load() }, [dueFilter, whose])

  const toggle = async (item) => {
    await fetch(`/api/commitments/${item.id}/status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: item.status === 'open' ? 'done' : 'open' }),
    })
    await load()
  }

  if (!items) return <Spinner label="Loading commitments…" />
  const shown = filter === 'all' ? items
    : items.filter((i) => i.status === filter)
  const openCount = items.filter((i) => i.status === 'open').length
  const overdue = items.filter(
    (i) => i.status === 'open' && i.due_state === 'overdue').length
  const soon = items.filter(
    (i) => i.status === 'open' && i.due_state === 'due-soon').length
  return (
    <section>
      <div className="section-head">
        <h2>Commitments</h2>
        <span className="muted">
          {openCount} open of {items.length}
          {overdue > 0 && (
            <span className="topic-chip due-overdue">{overdue} overdue</span>
          )}
          {soon > 0 && (
            <span className="topic-chip due-due-soon">{soon} due soon</span>
          )}
        </span>
      </div>
      <div className="toolbar">
        <label>
          Show
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="open">Open</option>
            <option value="done">Done</option>
            <option value="all">All</option>
          </select>
        </label>
        <label>
          When
          <select value={dueFilter}
            onChange={(e) => setDueFilter(e.target.value)}>
            <option value="">Any date</option>
            <option value="overdue">Overdue</option>
            <option value="due-soon">Due soon</option>
          </select>
        </label>
        <label>
          Whose
          <select value={whose} onChange={(e) => setWhose(e.target.value)}>
            <option value="">Everyone</option>
            <option value="mine">Mine</option>
            <option value="theirs">Theirs</option>
          </select>
        </label>
      </div>
      {shown.length === 0 && <p className="muted">Nothing here.</p>}
      <ul className="list">
        {shown.map((item) => (
          <li key={item.id}
            className={`commitment ${item.status} due-${item.due_state}`}>
            <label className="commit-row">
              <input
                type="checkbox"
                checked={item.status === 'done'}
                onChange={() => toggle(item)}
                aria-label={`Mark ${item.task} ${item.status === 'open' ? 'done' : 'open'}`}
              />
              <span className="commit-body">
                <span className="commit-task">
                  <strong>{item.owner}</strong>: {item.task}
                  {item.due_date ? <span className="due"> due {item.due_date}</span> : null}
                  {dueChip(item)}
                </span>
                <button className="link" onClick={(e) => {
                  e.preventDefault()
                  onOpenRecording(item.recording_id)
                }}>
                  {item.meeting_title || 'Untitled meeting'}
                  {item.segment_start != null ? ` · at ${formatDuration(item.segment_start)}` : ''}
                </button>
              </span>
            </label>
          </li>
        ))}
      </ul>
    </section>
  )
}

const EVENT_LABELS = {
  imported: 'Imported',
  captured: 'Captured',
  deleted: 'Deleted',
  'sync-skipped-as-deleted': 'Sync blocked',
  retried: 'Retried',
  resynced: 'Re-synced',
  reprocessed: 'Reprocessed',
}

function activityDescription(e) {
  const c = e.context || {}
  if (e.event_type === 'imported') {
    return `${c.original_filename || e.stem}${c.source_format ? ` (${c.source_format})` : ''}`
  }
  if (e.event_type === 'captured') return c.file || e.stem
  if (e.event_type === 'deleted') {
    const bits = [c.title || e.stem]
    if (c.how) bits.push(`(${c.how})`)
    if (c.context) bits.push(`— ${c.context}`)
    return bits.join(' ')
  }
  if (e.event_type === 'sync-skipped-as-deleted') {
    return `${e.stem} — kept out (previously deleted)`
  }
  if (e.event_type === 'retried' || e.event_type === 'resynced'
      || e.event_type === 'reprocessed') {
    const bits = [e.stem]
    if (c.scope) bits.push(`(${c.scope})`)
    if (c.outcome && c.outcome !== 'done') bits.push(`: ${c.outcome}`)
    return bits.join(' ')
  }
  return e.stem
}

// Read-only provenance log: a paper trail of imports, captures, and
// deletions. No management here by design.
function Activity() {
  const [events, setEvents] = useState(null)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    const q = filter ? `?type=${encodeURIComponent(filter)}` : ''
    fetch(`/api/activity${q}`).then((r) => r.json()).then(setEvents)
  }, [filter])

  return (
    <section>
      <div className="section-head">
        <h2>Activity</h2>
        <span className="muted">A history of what entered and left the archive.</span>
      </div>
      <div className="toolbar">
        <label>
          Type
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="">All</option>
            {Object.entries(EVENT_LABELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      {events == null ? (
        <Spinner label="Loading activity…" />
      ) : events.length === 0 ? (
        <p className="muted">No activity yet.</p>
      ) : (
        <ul className="list">
          {events.map((e) => (
            <li key={e.id} className="activity-item">
              <span className="activity-type">
                {EVENT_LABELS[e.event_type] || e.event_type}
              </span>
              <span className="activity-what">{activityDescription(e)}</span>
              <span className="activity-when">{formatDate(e.created_at)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

const SORTS = {
  newest: { label: 'Newest first', fn: (a, b) => recordingDate(b).localeCompare(recordingDate(a)) },
  oldest: { label: 'Oldest first', fn: (a, b) => recordingDate(a).localeCompare(recordingDate(b)) },
  longest: { label: 'Longest first', fn: (a, b) => (b.duration_seconds || 0) - (a.duration_seconds || 0) },
  shortest: { label: 'Shortest first', fn: (a, b) => (a.duration_seconds || 0) - (b.duration_seconds || 0) },
  title: { label: 'Title A to Z', fn: (a, b) => (a.title || a.file_name).localeCompare(b.title || b.file_name) },
}

// Voice-Memos-style scrolling waveform: recent level history as vertical
// bars, newest at the right, mic (You) mirrored above the center line and
// system (Meeting) below, so a dead source shows as a bald half. With
// `single` (in-person capture) there is no system half: the mic draws
// symmetrically around the center line on its own.
function Waveform({ historyRef, single }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    // Environments without a canvas implementation (jsdom under test)
    // hand back null; the meters are decoration, so skip drawing
    // rather than throwing from an animation frame.
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const reduce = window.matchMedia
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf, last = 0
    const draw = (t) => {
      raf = requestAnimationFrame(draw)
      // Reduced motion: slow the refresh instead of removing it.
      if (reduce && t - last < 200) return
      last = t
      const cs = getComputedStyle(document.documentElement)
      const accent = cs.getPropertyValue('--accent').trim() || '#2f6fe0'
      const muted = cs.getPropertyValue('--muted').trim() || '#888'
      const line = cs.getPropertyValue('--border-strong').trim() || '#ccc'
      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth || 300
      const h = canvas.clientHeight || 64
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr; canvas.height = h * dpr
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      const mid = h / 2
      const hist = historyRef.current || []
      // One bar per two 10Hz samples (0.2s each, louder of the pair) at a
      // 3px step, so a typical bar width keeps 40-55s of history visible.
      const step = 3
      ctx.fillStyle = line
      ctx.fillRect(0, mid - 0.5, w, 1)
      for (let i = hist.length - 1, bar = 1; i >= 0; i -= 2, bar++) {
        const x = w - bar * step
        if (x < -step) break
        const a = hist[i], b = hist[i - 1] || a
        const mic = Math.max(a.mic || 0, b.mic || 0)
        const micH = Math.min(mid, Math.sqrt(mic) * mid * 1.8)
        ctx.fillStyle = accent
        ctx.fillRect(x, mid - micH, 2, micH)
        if (single) {
          ctx.fillRect(x, mid, 2, micH)
        } else {
          const sys = Math.max(a.system || 0, b.system || 0)
          const sysH = Math.min(mid, Math.sqrt(sys) * mid * 1.8)
          ctx.fillStyle = muted
          ctx.fillRect(x, mid, 2, sysH)
        }
      }
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [historyRef, single])
  return <canvas ref={canvasRef} className="waveform" aria-hidden="true" />
}

// "Record meeting" with a capture-mode choice: an online call gets the
// two-channel capture (mic = you, system = everyone else); an in-person
// meeting records the mic only, so every voice in the room is diarized
// normally instead of being labeled as you. Remembers the last mode.
function RecordMenu({ onStart }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const last = localStorage.getItem('otterRecordMode') || 'online'
  // How many people are in the room, remembered between recordings.
  const [people, setPeople] = useState(
    () => localStorage.getItem('otterInPersonSpeakers') || '')
  useEffect(() => {
    if (!open) return
    const close = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])
  const pick = (mode) => () => {
    setOpen(false)
    // The room's speaker count only means something in person; an
    // online capture already separates the two sides by channel.
    const n = parseInt(people, 10)
    const valid = Number.isInteger(n) && n >= 1 && n <= 20
    if (mode === 'in_person' && valid) {
      localStorage.setItem('otterInPersonSpeakers', String(n))
    }
    onStart(mode, mode === 'in_person' && valid ? n : undefined)
  }
  const item = (mode, label, hint) => (
    <button role="menuitem" onClick={pick(mode)}>
      {label}{last === mode ? ' ✓' : ''}
      <span className="menu-hint">{hint}</span>
    </button>
  )
  return (
    <span className="row-menu record-menu" ref={ref}>
      <button className="ghost" aria-haspopup="menu" aria-expanded={open}
        title="Record a meeting without a bot. Everything stays on this Mac."
        onClick={() => setOpen(!open)}>
        Record meeting
      </button>
      {open && (
        <span className="menu" role="menu">
          {item('online', 'Online meeting',
            'Captures the call audio and your mic; your lines are labeled as you.')}
          {item('in_person', 'In person',
            'Mic only; every voice in the room is identified by speaker.')}
          <label className="record-people">
            People in the room
            <input type="number" min="1" max="20" placeholder="?"
              aria-label="People in the room"
              value={people}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => setPeople(e.target.value)} />
            <span className="menu-hint">
              Optional, and only used in person. Speaker detection guesses
              this badly in a noisy room.
            </span>
          </label>
        </span>
      )}
    </span>
  )
}

// The one home for recording state: a floating island pinned bottom-center
// above all content, on every view. Collapsed it is a compact pill; a
// click expands it into a focus view with the big waveform and a notes
// box. Expanding and collapsing is pure UI; the capture itself runs in
// the backend helper and is never interrupted.
function RecordingIsland({ recStatus, recLevels, historyRef, onStop,
                           notes, onNotes, notesState }) {
  const [open, setOpen] = useState(false)
  const active = recStatus?.state === 'recording'
    || recStatus?.state === 'starting'
  useEffect(() => { if (!active) setOpen(false) }, [active])
  useEffect(() => {
    if (!open) return
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open])
  if (!active) return null
  const inPerson = recStatus.mode === 'in_person'
  const warning = (recLevels?.warnings || [])[0]

  if (!open) {
    return (
      <div className="island" role="status">
        <button className="island-body" aria-expanded={false}
          aria-label="Expand recording view"
          onClick={() => setOpen(true)}>
          <span className="rec-dot" aria-hidden="true" />
          <span className="rec-elapsed">
            {recStatus.state === 'starting'
              ? 'Starting…' : formatDuration(recStatus.seconds)}
          </span>
          <span className="island-wave">
            <Waveform historyRef={historyRef} single={inPerson} />
          </span>
          {warning && (
            <span className="island-warn" title={warning.message}>⚠</span>
          )}
        </button>
        <button className="primary rec-stop" onClick={onStop}>Stop</button>
      </div>
    )
  }
  return (
    <>
      <div className="island-backdrop" onClick={() => setOpen(false)} />
      <div className="island island-expanded" role="dialog"
        aria-label="Recording">
        <div className="island-head">
          <span className="rec-dot" aria-hidden="true" />
          <span className="island-mode">
            {inPerson ? 'In person' : 'Online meeting'}
          </span>
          <span className="island-timer">
            {recStatus.state === 'starting'
              ? 'Starting…' : formatDuration(recStatus.seconds)}
          </span>
        </div>
        <div className="island-wave-large">
          {!inPerson && (
            <span className="wave-labels">
              <span>You</span><span>Meeting</span>
            </span>
          )}
          <Waveform historyRef={historyRef} single={inPerson} />
        </div>
        {warning && (
          <p className="rec-warning" role="alert">⚠ {warning.message}</p>
        )}
        <label className="island-notes">
          My notes
          <textarea
            value={notes}
            onChange={(e) => onNotes(e.target.value)}
            placeholder="Jot as you listen. These attach to the recording when it finishes processing."
          />
          <span className="muted note-state" role="status">{notesState}</span>
        </label>
        <div className="island-actions">
          <button className="primary rec-stop" onClick={onStop}>Stop</button>
          <button className="ghost" onClick={() => setOpen(false)}>
            Collapse
          </button>
        </div>
      </div>
    </>
  )
}

// Expand-in-place preview shown under a list row, without leaving the list.
// The folder digest: one synthesized, longitudinal view of everything
// filed in a folder. Generated on demand and cached server-side; when
// the folder grows, only the new recordings are sent to the model.
function FolderDigest({ folder, onBack, onOpenRecording }) {
  const [state, setState] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState(null)
  const [q, setQ] = useState('')
  const [asking, setAsking] = useState(false)
  const [answer, setAnswer] = useState(null)

  const load = async () => {
    try {
      const res = await fetch(`/api/folders/${folder.id}/digest`)
      if (!res.ok) { setError('Could not load this digest'); return }
      setState(await res.json())
    } catch { setError('Could not load this digest') }
  }

  useEffect(() => { load() }, [folder.id])

  const build = async (force) => {
    setBusy(force ? 'rebuild' : 'build')
    setError(null)
    try {
      const res = await fetch(`/api/folders/${folder.id}/digest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: !!force }),
      })
      const body = await res.json()
      if (!res.ok) setError(body.detail || 'Could not build this digest')
      else setState(body)
    } catch {
      setError('Could not build this digest')
    } finally {
      setBusy('')
    }
  }

  const togglePrivileged = async (on) => {
    const ok = window.confirm(on
      ? `Mark the folder "${folder.name}" privileged?\n\n`
        + 'WILL CHANGE: every recording in it is marked privileged now, '
        + 'and anything filed here later is marked as it arrives. Its '
        + 'digest, summaries, and questions about this folder run on the '
        + 'local model, with no cloud calls.\n\n'
        + 'WILL NOT CHANGE: nothing is deleted, and summaries already '
        + 'written stay as they are until they are regenerated.'
      : `Stop marking new arrivals in "${folder.name}" privileged?\n\n`
        + 'WILL CHANGE: recordings filed here from now on are not marked.'
        + '\n\nWILL NOT CHANGE: recordings already marked privileged stay '
        + 'privileged, so this folder still stays local until you unmark '
        + 'them one by one.')
    if (!ok) return
    setError(null)
    try {
      const res = await fetch(`/api/folders/${folder.id}/privileged`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ privileged: on }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body.detail || 'Could not change this folder')
        return
      }
      await load()
    } catch { setError('Could not change this folder') }
  }

  const ask = async (e) => {
    e.preventDefault()
    if (!q.trim()) return
    setAsking(true)
    setAnswer(null)
    try {
      const res = await fetch(`/api/ask?q=${encodeURIComponent(q)}`
        + `&folder_id=${folder.id}`)
      const body = await res.json()
      setAnswer(res.ok ? body
        : { answer: body.detail || 'Could not answer that', sources: [] })
    } catch {
      setAnswer({ answer: 'Could not answer that', sources: [] })
    } finally {
      setAsking(false)
    }
  }

  if (!state) {
    return (
      <article className="folder-digest">
        <button className="back" onClick={onBack}>&larr; Recordings</button>
        {error ? <p className="error-msg" role="alert">{error}</p>
          : <Spinner label="Loading digest…" />}
      </article>
    )
  }

  const stored = state.digest
  const d = stored && stored.digest
  const plural = (n) => (n === 1 ? '' : 's')

  return (
    <article className="folder-digest">
      <button className="back" onClick={onBack}>&larr; Recordings</button>
      <div className="section-head">
        <h2>{state.folder.name} digest</h2>
        <span className="digest-actions">
          {stored && (
            <button className="link" onClick={() => build(true)}
              disabled={!!busy}
              title="Ignore the stored digest and read every recording again.">
              {busy === 'rebuild' ? 'Rebuilding…' : 'Rebuild from scratch'}
            </button>
          )}
          <button className={stored ? 'ghost' : 'primary'}
            onClick={() => build(false)} disabled={!!busy}>
            {busy === 'build'
              ? (stored ? 'Updating…' : 'Generating…')
              : (stored ? 'Update' : 'Generate digest')}
          </button>
        </span>
      </div>

      {stored ? (
        <p className="digest-meta muted">
          Generated from {stored.recordings_count} recording
          {plural(stored.recordings_count)}, last updated{' '}
          {formatWhen(stored.updated_at)}
          {state.privileged && <> · local model only</>}
        </p>
      ) : (
        <p className="digest-meta muted">
          No digest yet. {state.recordings_in_folder} recording
          {plural(state.recordings_in_folder)} in this folder.
        </p>
      )}

      {stored && stored.new_since > 0 && (
        <p className="digest-stale" role="status">
          {stored.new_since} recording{plural(stored.new_since)} added since
          this digest. Update folds {stored.new_since === 1 ? 'it' : 'them'} in
          without rereading the rest.
        </p>
      )}
      {stored && stored.stale && stored.new_since === 0 && (
        <p className="digest-stale" role="status">
          This folder changed since the digest was written.
        </p>
      )}

      <label className="digest-privileged">
        <input type="checkbox" checked={state.folder.privileged}
          onChange={(e) => togglePrivileged(e.target.checked)} />
        <span>
          Keep this folder local.
          <span className="muted">
            {' '}Everything filed here is analyzed on the local model, and
            this digest never leaves the machine.
          </span>
        </span>
      </label>

      {error && <p className="error-msg" role="alert">{error}</p>}

      <form className="digest-ask" onSubmit={ask} role="search">
        <input value={q} onChange={(e) => setQ(e.target.value)}
          aria-label={`Ask about ${state.folder.name}`}
          placeholder={`Ask a question about ${state.folder.name}…`} />
        <button type="submit" className="ghost" disabled={asking}>
          {asking ? 'Asking…' : 'Ask this folder'}
        </button>
      </form>
      {answer && (
        <section className="answer">
          <div className="section-head">
            <h3>Answer</h3>
            <button className="link" onClick={() => setAnswer(null)}>Clear</button>
          </div>
          <p>{answer.answer}</p>
          {(answer.sources || []).length > 0 && (
            <p className="sources">
              {answer.sources.map((s) => (
                <button key={s.recording_id} className="link source"
                  onClick={() => onOpenRecording(s.recording_id)}>
                  [{s.n}] {s.title || 'Untitled'}
                </button>
              ))}
            </p>
          )}
        </section>
      )}

      {!d ? (
        <p className="muted">
          Nothing has been synthesized yet. Generate a digest to see the
          themes, changes, and open threads across this folder.
        </p>
      ) : (
        <div className="digest-body">
          <section className="summary">
            <h3>Over time</h3>
            <p>{d.overview}</p>
          </section>

          {d.themes.length > 0 && (
            <section className="summary">
              <h3>Recurring themes</h3>
              <ul className="digest-themes">
                {d.themes.map((t, i) => (
                  <li key={i}>
                    <strong>{t.theme}</strong>
                    {t.first_appeared && (
                      <span className="muted"> · first appeared{' '}
                        {formatDate(t.first_appeared)}</span>
                    )}
                    {(t.framing_then || t.framing_now) && (
                      <div className="digest-framing">
                        <span>{t.framing_then}</span>
                        <span aria-hidden="true"> → </span>
                        <span>{t.framing_now}</span>
                      </div>
                    )}
                    {t.how_it_changed && <div>{t.how_it_changed}</div>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {d.shifts.length > 0 && (
            <section className="summary">
              <h3>Notable shifts</h3>
              <ul>
                {d.shifts.map((s, i) => (
                  <li key={i}>
                    {s.date && <strong>{formatDate(s.date)}: </strong>}
                    {s.shift}
                    {s.note && <span className="muted"> ({s.note})</span>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {d.open_threads.length > 0 && (
            <section className="summary">
              <h3>Raised and never resolved</h3>
              <ul>
                {d.open_threads.map((t, i) => (
                  <li key={i}>
                    {t.thread}
                    {t.raised_on && (
                      <span className="muted"> · raised{' '}
                        {formatDate(t.raised_on)}</span>
                    )}
                    {t.last_mentioned && (
                      <span className="muted"> · last mentioned{' '}
                        {formatDate(t.last_mentioned)}</span>
                    )}
                    {t.note && <div className="muted">{t.note}</div>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {d.commitments.length > 0 && (
            <section className="summary">
              <h3>Commitments across this folder</h3>
              <ul className="actions">
                {d.commitments.map((c, i) => (
                  <li key={i}>
                    <strong>{c.owner}</strong>: {c.commitment}
                    <span className={`topic-chip status-${c.status}`}>
                      {c.status}
                    </span>
                    {c.made_on && (
                      <span className="muted"> · {formatDate(c.made_on)}</span>
                    )}
                    {c.note && <div className="muted">{c.note}</div>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {d.quotes.length > 0 && (
            <section className="summary">
              <h3>In their own words</h3>
              <ul className="digest-quotes">
                {d.quotes.map((qt, i) => (
                  <li key={i}>
                    <blockquote>“{qt.quote}”</blockquote>
                    <button className="link source"
                      onClick={() => onOpenRecording(qt.recording_id)}>
                      {qt.speaker || 'Unknown'} · {qt.timestamp}
                    </button>
                    {qt.why && <span className="muted"> ({qt.why})</span>}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </article>
  )
}

function RowPreview({ recordingId, onOpenFull }) {
  const hueFor = useChipStyle()
  const [d, setD] = useState(null)
  useEffect(() => {
    let active = true
    fetch(`/api/recordings/${recordingId}`).then((r) => r.json())
      .then((body) => { if (active) setD(body) })
    return () => { active = false }
  }, [recordingId])
  if (!d) return <div className="row-preview"><Spinner label="Loading…" /></div>
  const s = d.summary
  const speakers = [...new Set((d.segments || [])
    .map((x) => x.speaker).filter(Boolean))]
  return (
    <div className="row-preview">
      <audio controls preload="metadata" className="player"
        src={`/api/recordings/${recordingId}/audio`} data-testid="preview-audio" />
      <div className="preview-chips">
        {s?.action_items?.length > 0 && (
          <span className="topic-chip">{s.action_items.length} action item{s.action_items.length === 1 ? '' : 's'}</span>
        )}
        {speakers.length > 0 && (
          <span className="topic-chip">{speakers.length} speaker{speakers.length === 1 ? '' : 's'}</span>
        )}
        {d.folder && <span className="topic-chip folder-chip">{d.folder}</span>}
      </div>
      {s?.summary && <p className="preview-summary">{s.summary}</p>}
      {(d.segments || []).slice(0, 4).map((seg, i) => (
        <p key={i} className="preview-line">
          <span className="speaker mini"
            style={hueFor(seg.speaker)}>
            {seg.speaker || 'unknown'}
          </span>
          <span className="preview-text">{seg.text}</span>
        </p>
      ))}
      <button className="link" onClick={onOpenFull}>
        Open full transcript and notes →
      </button>
    </div>
  )
}

export default function App() {
  const [recordings, setRecordings] = useState(null)
  const [selected, setSelected] = useState(null)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [syncing, setSyncing] = useState(false)
  const [syncMsg, setSyncMsg] = useState(null)
  const [sort, setSort] = useState('newest')
  const [speaker, setSpeaker] = useState('')
  const [topic, setTopic] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [view, setView] = useState('recordings')
  const [person, setPerson] = useState(null)
  const [calStatus, setCalStatus] = useState(null)
  const [calMsg, setCalMsg] = useState(null)

  const [selectedIds, setSelectedIds] = useState(new Set())
  const [actionMsg, setActionMsg] = useState(null)
  const [folders, setFolders] = useState([])
  const [folderFilter, setFolderFilter] = useState('')
  const [searchedIn, setSearchedIn] = useState(null)
  // The folder whose digest is open, if any. Digests live in the
  // recordings view, in place of the list.
  const [digestFolder, setDigestFolder] = useState(null)
  const [suggesting, setSuggesting] = useState(false)
  const [expandedId, setExpandedId] = useState(null)

  // Three-pane mode on wide displays: the recordings list narrows to a
  // scannable column and the selected recording shows in a persistent
  // detail pane instead of navigating. Kept in sync with the same
  // 1300px breakpoint the stylesheet uses.
  const [wide, setWide] = useState(() =>
    typeof window.matchMedia === 'function'
    && window.matchMedia('(min-width: 1300px)').matches)
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const mq = window.matchMedia('(min-width: 1300px)')
    const onChange = (e) => setWide(e.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  // Expand-in-place exists only below the wide breakpoint, so there are
  // never two ways to view the same recording at once.
  useEffect(() => { if (wide) setExpandedId(null) }, [wide])

  // Draggable pane widths, remembered across reloads. Applied as CSS
  // variables that only the wide (1300px+) stylesheet consumes, so
  // narrower layouts are untouched.
  const [paneW, setPaneW] = useState(() => {
    try {
      return { ...PANE_DEFAULTS,
        ...JSON.parse(localStorage.getItem('otterPaneWidths') || '{}') }
    } catch { return { ...PANE_DEFAULTS } }
  })
  const setPane = (key, px) => {
    const [min, max] = PANE_LIMITS[key]
    const v = Math.round(Math.min(max, Math.max(min, px)))
    setPaneW((prev) => {
      if (prev[key] === v) return prev
      const next = { ...prev, [key]: v }
      localStorage.setItem('otterPaneWidths', JSON.stringify(next))
      return next
    })
  }

  // Per-person chip colors: loaded once, provided app-wide, updated
  // optimistically so every chip repaints the moment a color is picked.
  const [speakerColors, setSpeakerColors] = useState({})
  useEffect(() => {
    fetch('/api/people/colors').then((r) => r.json())
      .then((body) => setSpeakerColors(body && typeof body === 'object' ? body : {}))
      .catch(() => {})
  }, [])
  const setSpeakerColor = async (name, color) => {
    setSpeakerColors((prev) => {
      const next = { ...prev }
      if (color && color !== 'auto') next[name] = color
      else delete next[name]
      return next
    })
    await fetch(`/api/people/${encodeURIComponent(name)}/color`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ color: color === 'auto' ? null : color }),
    })
  }
  const hueFor = (name) => chipStyle(speakerColors, name)

  // Ring buffer of recent level samples for the live waveform.
  const levelHistory = useRef([])

  // Notes jotted in the recording island. The draft persists locally
  // (the recording has no id until processing finishes) with the same
  // 800ms debounce as the recording page, and attaches to the finished
  // recording the moment it appears in the list.
  const [islandNotes, setIslandNotes] = useState(
    () => localStorage.getItem('otterIslandNotes') || '')
  const [islandNotesState, setIslandNotesState] = useState('')
  useEffect(() => {
    if (islandNotes === (localStorage.getItem('otterIslandNotes') || '')) {
      return
    }
    setIslandNotesState('Saving…')
    const t = setTimeout(() => {
      localStorage.setItem('otterIslandNotes', islandNotes)
      setIslandNotesState('Saved')
    }, 800)
    return () => clearTimeout(t)
  }, [islandNotes])

  const loadFolders = () =>
    fetch('/api/folders')
      .then((r) => r.json())
      .then((body) => setFolders(Array.isArray(body) ? body : []))

  useEffect(() => { loadFolders() }, [])

  // Escape collapses an expanded row preview.
  useEffect(() => {
    if (expandedId == null) return
    const onKey = (e) => { if (e.key === 'Escape') setExpandedId(null) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [expandedId])

  useEffect(() => {
    fetch('/api/calendar/status').then((r) => r.json()).then(setCalStatus)
  }, [])

  const connectCalendar = async () => {
    setCalMsg(null)
    const res = await fetch('/api/calendar/connect', { method: 'POST' })
    const body = await res.json()
    if (!res.ok) {
      setCalMsg(body.detail || 'Connection failed')
    } else {
      setCalMsg('Google Calendar connected')
      const m = await fetch('/api/calendar/match', { method: 'POST' })
      const mb = await m.json()
      setCalMsg(`Google Calendar connected · ${mb.matched} recordings matched`)
      fetch('/api/calendar/status').then((r) => r.json()).then(setCalStatus)
      refresh()
    }
  }

  const refresh = () =>
    fetch('/api/recordings').then((r) => r.json()).then(setRecordings)

  useEffect(() => {
    refresh()
    // Light polling so recordings the watcher is still processing
    // update their status without a manual reload.
    const timer = setInterval(refresh, 5000)
    return () => clearInterval(timer)
  }, [])

  const [recStatus, setRecStatus] = useState(null)
  const [recLevels, setRecLevels] = useState(null)

  const loadRecStatus = () =>
    fetch('/api/record/status').then((r) => r.json()).then(setRecStatus)

  useEffect(() => { loadRecStatus() }, [])

  // Poll while a recording is running so the elapsed time ticks.
  useEffect(() => {
    if (recStatus?.state === 'recording' || recStatus?.state === 'starting') {
      const t = setTimeout(loadRecStatus, 2000)
      return () => clearTimeout(t)
    }
  }, [recStatus])

  // Poll input levels ~10x/second while recording, feeding both the
  // status and the scrolling waveform's level history.
  const recording = recStatus?.state === 'recording'
  useEffect(() => {
    if (!recording) { setRecLevels(null); levelHistory.current = []; return }
    let active = true
    const tick = async () => {
      try {
        const r = await fetch('/api/record/levels')
        const body = await r.json()
        if (!active) return
        setRecLevels(body)
        // Keep ~60s at 10Hz (600 samples), newest last.
        const hist = levelHistory.current
        hist.push({ mic: body.mic || 0, system: body.system || 0 })
        if (hist.length > 600) hist.splice(0, hist.length - 600)
      } catch { /* transient */ }
    }
    tick()
    const timer = setInterval(tick, 100)
    return () => { active = false; clearInterval(timer) }
  }, [recording])

  // Milestone C: offer to record a meeting that is starting now. Never
  // auto-records; the prompt is dismissible and won't nag for the same
  // event twice.
  const [startingEvent, setStartingEvent] = useState(null)
  const [dismissedEvent, setDismissedEvent] = useState(null)
  useEffect(() => {
    let active = true
    const poll = async () => {
      try {
        const r = await fetch('/api/calendar/starting')
        const body = await r.json()
        if (active) setStartingEvent(body.event)
      } catch { /* ignore */ }
    }
    poll()
    const timer = setInterval(poll, 30000)
    return () => { active = false; clearInterval(timer) }
  }, [])

  const startRecording = async (mode = 'online', numSpeakers = undefined) => {
    if (!localStorage.getItem('otterConsentAck')) {
      if (!window.confirm(
        'One-time reminder before your first recording:\n\n' +
        'Rules about recording calls vary by place, and many places ' +
        'require the other side’s consent. Make sure everyone on ' +
        'the call is okay with being recorded.\n\n' +
        'This reminder will not be shown again.')) return
      localStorage.setItem('otterConsentAck', '1')
    }
    localStorage.setItem('otterRecordMode', mode)
    const res = await fetch('/api/record/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        numSpeakers ? { mode, num_speakers: numSpeakers } : { mode }),
    })
    const body = await res.json()
    if (!res.ok) {
      setSyncMsg(body.detail || 'Recorder could not start')
    }
    await loadRecStatus()
  }

  const stopRecording = async () => {
    const res = await fetch('/api/record/stop', { method: 'POST' })
    const body = await res.json()
    if (res.ok) {
      setSyncMsg(`Saved ${body.file}, processing…`)
      // Hand the island jottings off: they attach to this file's
      // recording as soon as processing lands it in the list.
      if (islandNotes.trim() && body.file) {
        localStorage.setItem('otterPendingNotes', JSON.stringify(
          { file: body.file, text: islandNotes }))
      }
      localStorage.removeItem('otterIslandNotes')
      setIslandNotes('')
      setIslandNotesState('')
      refresh()
    } else {
      setSyncMsg(body.detail || 'Stop failed')
    }
    await loadRecStatus()
  }

  // Attach island notes to the finished recording once it exists.
  useEffect(() => {
    if (!recordings) return
    const raw = localStorage.getItem('otterPendingNotes')
    if (!raw) return
    let pending
    try { pending = JSON.parse(raw) } catch {
      localStorage.removeItem('otterPendingNotes')
      return
    }
    const match = recordings.find((r) => r.file_name === pending.file)
    if (!match) return
    localStorage.removeItem('otterPendingNotes')
    fetch(`/api/recordings/${match.id}/notes`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: pending.text }),
    }).then((res) => {
      if (res.ok) setActionMsg('Your notes were attached to the new recording')
    })
  }, [recordings])

  const syncFromIphone = async () => {
    setSyncing(true)
    setSyncMsg(null)
    try {
      const res = await fetch('/api/sync', { method: 'POST' })
      const body = await res.json()
      if (!res.ok) {
        setSyncMsg(body.detail || 'Sync failed')
      } else if (body.copied === 0) {
        setSyncMsg('Nothing new')
      } else {
        setSyncMsg(`${body.copied} new recording${body.copied === 1 ? '' : 's'} syncing…`)
        refresh()
      }
    } catch {
      setSyncMsg('Sync failed')
    } finally {
      setSyncing(false)
    }
  }

  const [askResult, setAskResult] = useState(null)
  const [asking, setAsking] = useState(false)

  // A search made while the list is filtered to a folder means that
  // folder. Both halves of the field follow the filter, and the scope
  // is stated on screen with one click to widen it.
  const scopedFolder =
    folderFilter && folderFilter !== 'unfiled'
      ? folders.find((f) => String(f.id) === folderFilter) || null
      : null

  const ask = async (scope = scopedFolder) => {
    if (!query.trim()) return
    setAsking(true)
    setAskResult(null)
    try {
      const suffix = scope ? `&folder_id=${scope.id}` : ''
      const res = await fetch(
        `/api/ask?q=${encodeURIComponent(query)}${suffix}`)
      setAskResult(await res.json())
    } finally {
      setAsking(false)
    }
  }

  // One field does both: every submit runs keyword search immediately;
  // a question-shaped query also fires Ask in parallel, its answer card
  // landing above the results when it returns. Short keyword queries
  // never spend an LLM call; the "Ask this instead" link forces one
  // when the detection guesses wrong.
  const search = async (e, scope = scopedFolder) => {
    if (e) e.preventDefault()
    if (!query.trim()) {
      setResults(null); setAskResult(null); setSearchedIn(null); return
    }
    if (looksLikeQuestion(query)) {
      ask(scope)
    } else {
      setAskResult(null)
    }
    setSearchedIn(scope)
    const suffix = scope ? `&folder_id=${scope.id}` : ''
    const res = await fetch(
      `/api/search?q=${encodeURIComponent(query)}${suffix}`)
    setResults(await res.json())
  }

  // Widen a folder-scoped search to the whole archive without retyping.
  const searchEverything = () => {
    setFolderFilter('')
    search(null, null)
  }

  const allSpeakers = useMemo(() => {
    const s = new Set()
    for (const r of recordings || []) for (const sp of r.speakers || []) s.add(sp)
    return [...s].sort()
  }, [recordings])

  const allTopics = useMemo(() => {
    const counts = new Map()
    for (const r of recordings || []) {
      for (const t of r.topics || []) counts.set(t, (counts.get(t) || 0) + 1)
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([t]) => t)
  }, [recordings])

  const filtersActive = speaker || topic || dateFrom || dateTo || folderFilter

  const visible = useMemo(() => {
    if (!recordings) return null
    let list = recordings
    if (speaker) list = list.filter((r) => (r.speakers || []).includes(speaker))
    if (topic) list = list.filter((r) => (r.topics || []).includes(topic))
    if (dateFrom) list = list.filter((r) => recordingDate(r) >= dateFrom)
    if (dateTo) list = list.filter((r) => recordingDate(r) <= dateTo)
    if (folderFilter === 'unfiled') list = list.filter((r) => !r.folder_id)
    else if (folderFilter) {
      list = list.filter((r) => String(r.folder_id) === folderFilter)
    }
    return [...list].sort(SORTS[sort].fn)
  }, [recordings, speaker, topic, dateFrom, dateTo, sort, folderFilter])

  // Three-pane keyboard navigation: up/down moves the selection through
  // the visible list, without stealing arrows from form fields.
  useEffect(() => {
    if (!wide || view !== 'recordings') return
    const onKey = (e) => {
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
      const t = e.target
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
        || t.tagName === 'SELECT' || t.isContentEditable)) return
      if (!visible || visible.length === 0) return
      e.preventDefault()
      const idx = visible.findIndex((r) => r.id === selected)
      const next = e.key === 'ArrowDown'
        ? Math.min(visible.length - 1, idx + 1)
        : Math.max(0, idx - 1)
      setSelected(visible[next].id)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [wide, view, visible, selected])

  // Keep the keyboard-selected row in view inside the list column.
  useEffect(() => {
    if (!wide || selected == null) return
    const row = document.querySelector('.list-pane .dense-item.selected')
    if (row && typeof row.scrollIntoView === 'function') {
      row.scrollIntoView({ block: 'nearest' })
    }
  }, [wide, selected])

  const pendingSuggestions = (recordings || [])
    .filter((r) => r.suggested_folder && !r.folder_id).length

  const processing = (recordings || []).filter(
    (r) => r.status === 'pending' || r.status === 'transcribing',
  ).length

  const toggleSelected = (id) =>
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const allVisibleSelected = visible != null && visible.length > 0 &&
    visible.every((r) => selectedIds.has(r.id))

  const selectAllVisible = () =>
    setSelectedIds(allVisibleSelected
      ? new Set() : new Set(visible.map((r) => r.id)))

  const clearSelection = () => setSelectedIds(new Set())

  const deleteOne = async (r) => {
    if (!window.confirm(deleteWarning(`"${r.title || r.file_name}"`))) return
    const res = await fetch(`/api/recordings/${r.id}`, { method: 'DELETE' })
    setActionMsg(res.ok ? 'Recording deleted' : 'Delete failed')
    refresh()
  }

  // Recovery actions run in the background; poll the job endpoint until
  // it settles, then refresh, so the UI never blocks (same idea as the
  // Reset speakers polling on the recording page).
  const watchJob = (id, label) => {
    const poll = async () => {
      try {
        const res = await fetch(`/api/recordings/${id}/job`)
        const body = await res.json()
        if (body.job?.state === 'running') {
          setTimeout(poll, 3000)
          return
        }
        if (body.job?.state === 'failed') {
          window.alert(`${label} failed: ${body.job.error || 'unknown error'}`)
        } else {
          setActionMsg(`${label} finished`)
        }
      } catch { /* transient; the next refresh shows the outcome */ }
      refresh()
    }
    setTimeout(poll, 3000)
  }

  const startRecovery = async (url, opts, label) => {
    const res = await fetch(url, { method: 'POST', ...opts })
    const body = await res.json()
    if (!res.ok) {
      window.alert(body.detail || `${label} could not start`)
      return
    }
    setActionMsg(`${label} started, running in the background…`)
    const id = parseInt(url.match(/recordings\/(\d+)\//)[1], 10)
    watchJob(id, label)
  }

  const retryOne = async (r) => {
    const stuck = r.status === 'transcribing'
    if (!window.confirm(
      `Retry processing "${r.title || r.file_name}"?\n\n` +
      (stuck
        ? 'This recording has been stuck part-way through processing, '
          + 'which happens when the app was closed mid-run. '
        : '')
      + 'The pipeline runs again on the existing audio file: transcription, '
      + 'speaker detection, AI summary, and search indexing. A '
      + (stuck ? 'stuck' : 'failed')
      + ' recording has no results yet, so there is nothing to lose.')) return
    await startRecovery(`/api/recordings/${r.id}/retry`, {}, 'Retry')
  }

  const resyncOne = async (r) => {
    if (!window.confirm(
      `Re-sync "${r.title || r.file_name}" from Voice Memos?\n\n` +
      'WILL CHANGE: the local audio file is deleted and replaced with a '
      + 'fresh copy from the Voice Memos folder, then the transcript, '
      + 'machine speaker labels, AI summary and action items, and the '
      + 'search index are rebuilt from it.\n\n'
      + 'WILL NOT CHANGE: your manual title, My notes and enhanced notes, '
      + 'folder, privileged flag, commitment checkboxes, speaker-tag '
      + 'rejections, and calendar match. Pinned lines and speaker names '
      + 'you set yourself are re-attached to the same moments in time.\n\n'
      + 'This is the fix for a file that went corrupt on this Mac but is '
      + 'fine on the iPhone. It takes a few minutes and runs in the '
      + 'background.')) return
    await startRecovery(`/api/recordings/${r.id}/resync`, {}, 'Re-sync')
  }

  const reprocessOne = async (r, scope) => {
    const name = `"${r.title || r.file_name}"`
    const dialogs = {
      everything:
        `Reprocess everything for ${name}?\n\n` +
        'WILL CHANGE: transcript text and timing, machine speaker labels, '
        + 'AI summary and action items, cached translations, and the '
        + 'search index are all rebuilt from the audio.\n\n'
        + 'WILL NOT CHANGE: your manual title, My notes and enhanced '
        + 'notes, folder, privileged flag, commitment checkboxes, '
        + 'speaker-tag rejections, and calendar match. Pinned lines and '
        + 'speaker names you set yourself are re-attached to the same '
        + 'moments in time.\n\n'
        + 'This takes a few minutes and runs in the background.',
      speakers:
        `Redo speakers for ${name}?\n\n` +
        'WILL CHANGE: machine speaker labels (SPEAKER_XX and auto-tags) '
        + 'on unpinned lines get fresh speaker detection, then enrolled '
        + 'voices are re-applied where they match.\n\n'
        + 'WILL NOT CHANGE: pinned lines, speaker names you set yourself, '
        + 'speaker-tag rejections, the transcript text, the AI summary, '
        + 'your notes, and everything else. The transcript is NOT '
        + 're-transcribed.\n\n'
        + 'This takes a few minutes and runs in the background.',
      notes:
        `Redo AI notes for ${name}?\n\n` +
        'WILL CHANGE: the AI summary and its action items are rewritten '
        + 'from the existing transcript; new action items may add '
        + 'commitments.\n\n'
        + 'WILL NOT CHANGE: your manual title, My notes and enhanced '
        + 'notes, existing commitments and their checkboxes, the '
        + 'transcript, speaker labels, and everything else. The '
        + 'transcript is NOT re-transcribed.\n\n'
        + 'Runs in the background.',
    }
    if (!window.confirm(dialogs[scope])) return

    // Speaker detection gets two optional aids. Notes-only never
    // touches the audio, so it is not asked.
    const opts = {}
    if (scope === 'speakers' || scope === 'everything') {
      const answer = window.prompt(
        'How many people speak in this recording?\n\n'
        + 'Speaker detection normally works this out on its own, and '
        + 'often does it well. Pinning the number can rescue a '
        + 'recording it got wrong, but it can also make a good result '
        + 'worse by forcing voices together, so compare the result '
        + 'against what you had.\n\n'
        + 'Leave empty to let it decide, which is the better default.',
        '')
      if (answer === null) return
      if (answer.trim()) {
        const n = parseInt(answer.trim(), 10)
        if (!Number.isInteger(n) || n < 1 || n > 20) {
          window.alert('That is not a speaker count between 1 and 20, '
            + 'so nothing was started.')
          return
        }
        opts.num_speakers = n
      }
      opts.denoise = window.confirm(
        'Clean the audio before detecting speakers?\n\n'
        + 'This runs a noise-reduction pass first, which can help on a '
        + 'restaurant or street recording. It can also make things '
        + 'worse by smoothing away the very differences between voices, '
        + 'so it is off unless you ask for it.\n\n'
        + 'OK to clean the audio first, Cancel to use it as recorded.')
    }
    await startRecovery(`/api/recordings/${r.id}/reprocess`, {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope, ...opts }),
    }, scope === 'everything' ? 'Reprocess'
      : scope === 'speakers' ? 'Speaker redo' : 'Notes redo')
  }

  // Folder ops. promptForFolder returns a folder id, null to unfile,
  // or undefined when the user cancelled.
  const promptForFolder = async () => {
    const names = folders.map((f) => f.name)
    const input = window.prompt(
      'Folder name (existing or new). Leave empty to unfile.'
      + (names.length ? `\nExisting: ${names.join(', ')}` : ''), '')
    if (input === null) return undefined
    const name = input.trim()
    if (!name) return null
    const existing = folders.find(
      (f) => f.name.toLowerCase() === name.toLowerCase())
    if (existing) return existing.id
    const res = await fetch('/api/folders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
    const body = await res.json()
    await loadFolders()
    return body.id
  }

  const renameRecording = async (r) => {
    const t = window.prompt(
      'Title for this recording (leave empty to go back to the '
      + 'automatic title):', r.title || '')
    if (t === null) return
    await fetch(`/api/recordings/${r.id}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: t }),
    })
    refresh()
  }

  const moveOne = async (r) => {
    const fid = await promptForFolder()
    if (fid === undefined) return
    await fetch(`/api/recordings/${r.id}/folder`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_id: fid }),
    })
    refresh()
    loadFolders()
  }

  const moveSelected = async () => {
    const fid = await promptForFolder()
    if (fid === undefined) return
    const res = await fetch('/api/recordings/bulk-folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: [...selectedIds], folder_id: fid }),
    })
    const body = await res.json()
    setActionMsg(`${body.filed} recording${body.filed === 1 ? '' : 's'} moved`)
    clearSelection()
    refresh()
    loadFolders()
  }

  const createFolder = async () => {
    const name = window.prompt('New folder name:', '')
    if (!name || !name.trim()) return
    await fetch('/api/folders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    })
    loadFolders()
  }

  const renameFolder = async (f) => {
    const name = window.prompt('Rename folder to:', f.name)
    if (!name || name === f.name) return
    const res = await fetch(`/api/folders/${f.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
    if (!res.ok) {
      const body = await res.json()
      setActionMsg(body.detail || 'Rename failed')
    }
    loadFolders()
    refresh()
  }

  const deleteFolder = async (f) => {
    if (!window.confirm(
      `Delete the folder "${f.name}"?\n\nIts recordings are NOT `
      + 'deleted; they just become unfiled.')) return
    await fetch(`/api/folders/${f.id}`, { method: 'DELETE' })
    if (String(f.id) === folderFilter) setFolderFilter('')
    if (digestFolder && digestFolder.id === f.id) setDigestFolder(null)
    loadFolders()
    refresh()
  }

  const suggestFolders = async () => {
    setSuggesting(true)
    try {
      const res = await fetch('/api/folders/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      const body = await res.json()
      const n = Object.keys(body.suggestions || {}).length
      setActionMsg(res.ok
        ? `${n} folder suggestion${n === 1 ? '' : 's'} ready`
        : body.detail || 'Suggestions failed')
      refresh()
    } finally {
      setSuggesting(false)
    }
  }

  const acceptAllSuggestions = async () => {
    const res = await fetch('/api/folders/accept-all', { method: 'POST' })
    const body = await res.json()
    setActionMsg(`${body.filed} recording${body.filed === 1 ? '' : 's'} filed`)
    refresh()
    loadFolders()
  }

  const acceptSuggestion = async (r) => {
    await fetch(`/api/recordings/${r.id}/suggestion/accept`,
      { method: 'POST' })
    refresh()
    loadFolders()
  }

  const dismissSuggestion = async (r) => {
    await fetch(`/api/recordings/${r.id}/suggestion/dismiss`,
      { method: 'POST' })
    refresh()
  }

  const deleteSelected = async () => {
    const ids = [...selectedIds]
    const what = `${ids.length} recording${ids.length === 1 ? '' : 's'}`
    if (!window.confirm(deleteWarning(what))) return
    const res = await fetch('/api/recordings/bulk-delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    })
    if (res.ok) {
      const body = await res.json()
      setActionMsg(`${body.deleted} recording${body.deleted === 1 ? '' : 's'} deleted`)
      clearSelection()
    } else {
      setActionMsg('Delete failed')
    }
    refresh()
  }

  const NAV = ['recordings', 'people', 'voices', 'commitments', 'activity']

  // The one home for recording state: everything recording-shaped lives
  // in the floating corner. While recording, the island; otherwise the
  // meeting-aware prompt floats as a small banner in the same spot.
  const recordingLayer = (
    <>
      {!(recStatus?.state === 'recording' || recStatus?.state === 'starting')
        && startingEvent && startingEvent.id !== dismissedEvent && (
        <div className="meeting-prompt island-banner" role="status">
          <span>
            <strong>{startingEvent.title}</strong> is starting
            {startingEvent.attendees?.length
              ? ` with ${startingEvent.attendees.join(', ')}` : ''}. Record it?
          </span>
          <span className="prompt-actions">
            <button className="primary" onClick={() => {
              setDismissedEvent(startingEvent.id)
              startRecording('online')
            }}>
              Record meeting
            </button>
            <button className="link"
              onClick={() => setDismissedEvent(startingEvent.id)}>
              Not now
            </button>
          </span>
        </div>
      )}
      <RecordingIsland recStatus={recStatus} recLevels={recLevels}
        historyRef={levelHistory} onStop={stopRecording}
        notes={islandNotes} onNotes={setIslandNotes}
        notesState={islandNotesState} />
    </>
  )

  return (
    <SpeakerColorsCtx.Provider
      value={{ colors: speakerColors, setColor: setSpeakerColor }}>
    <div className="app-shell"
      style={wide ? { '--sidebar-w': `${paneW.sidebar}px`,
                      '--list-w': `${paneW.list}px` } : undefined}>
      <aside className="app-sidebar">
        <Brand />
        <nav className="side-nav">
          {NAV.map((v) => (
            <button key={v}
              className={`side-tab ${view === v && (selected == null || (wide && v === 'recordings')) ? 'active' : ''}`}
              onClick={() => { setView(v); setSelected(null); setPerson(null) }}>
              {v[0].toUpperCase() + v.slice(1)}
            </button>
          ))}
        </nav>
        {view === 'recordings' && (selected == null || wide) && (
          <div className="sidebar-folders">
            <div className="section-head">
              <h3>Folders</h3>
              <button className="link" onClick={createFolder}
                title="Create a new folder">+ New</button>
            </div>
            <button
              className={`folder-row ${folderFilter === '' ? 'active' : ''}`}
              onClick={() => setFolderFilter('')}>
              <span className="folder-name">All</span>
              <span className="folder-count">{recordings ? recordings.length : 0}</span>
            </button>
            <button
              className={`folder-row ${folderFilter === 'unfiled' ? 'active' : ''}`}
              onClick={() => setFolderFilter('unfiled')}>
              <span className="folder-name">Unfiled</span>
              <span className="folder-count">
                {(recordings || []).filter((r) => !r.folder_id).length}
              </span>
            </button>
            {folders.map((f) => (
              <div key={f.id} className="folder-item">
                <button
                  className={`folder-row ${folderFilter === String(f.id) ? 'active' : ''}`}
                  onClick={() => setFolderFilter(String(f.id))}>
                  <span className="folder-name">{f.name}</span>
                  <span className="folder-count">{f.count}</span>
                </button>
                <span className="folder-tools">
                  <button className="link" aria-label={`Digest of ${f.name}`}
                    onClick={() => {
                      setView('recordings'); setSelected(null)
                      setDigestFolder({ id: f.id, name: f.name })
                    }}>Digest</button>
                  <button className="link" aria-label={`Rename ${f.name}`}
                    onClick={() => renameFolder(f)}>Rename</button>
                  <button className="link danger" aria-label={`Delete ${f.name}`}
                    onClick={() => deleteFolder(f)}>Delete</button>
                </span>
              </div>
            ))}
          </div>
        )}
        <div className="sidebar-foot">
          <ThemeToggle />
        </div>
      </aside>

      {wide && (
        <PaneHandle
          label="Resize sidebar"
          value={paneW.sidebar}
          min={PANE_LIMITS.sidebar[0]}
          max={PANE_LIMITS.sidebar[1]}
          onChange={(px) => setPane('sidebar', px)}
          onReset={() => setPane('sidebar', PANE_DEFAULTS.sidebar)}
        />
      )}

      <main className={`app-content ${view === 'recordings' ? 'rec-view' : ''}`}>
        {digestFolder && view === 'recordings' ? (
          <FolderDigest
            folder={digestFolder}
            onBack={() => setDigestFolder(null)}
            onOpenRecording={(id) => { setDigestFolder(null); setSelected(id) }}
          />
        ) : selected != null && !(wide && view === 'recordings') ? (
          <Detail
            recordingId={selected}
            onBack={() => setSelected(null)}
            onDeleted={() => {
              setSelected(null)
              setActionMsg('Recording deleted')
              refresh()
            }}
          />
        ) : view === 'people' ? (
          person ? (
            <Person name={person} onBack={() => setPerson(null)}
              onOpenRecording={setSelected} />
          ) : (
            <PeopleList onSelect={setPerson} />
          )
        ) : view === 'voices' ? (
          <Voices />
        ) : view === 'commitments' ? (
          <Commitments onOpenRecording={setSelected} />
        ) : view === 'activity' ? (
          <Activity />
        ) : (
          <>
            <div className="rec-toolbar">
              <form onSubmit={search} role="search">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={scopedFolder
                    ? `Search ${scopedFolder.name}, or ask a question in plain words…`
                    : 'Search keywords, or ask a question in plain words…'}
                  aria-label={scopedFolder
                    ? `Search ${scopedFolder.name}`
                    : 'Search transcripts'}
                />
                <button type="submit" className="primary">Search</button>
              </form>
              <span className="toolbar-actions">
                {recStatus?.state === 'error' && recStatus.error && (
                  <span className="sync-msg error-msg" role="alert">
                    {recStatus.error}
                  </span>
                )}
                {calStatus && !calStatus.google_connected && (
                  <button className="ghost" onClick={connectCalendar}
                    title="Match recordings to your calendar for auto-titles. Until connected, a sample calendar is used.">
                    Connect Calendar
                  </button>
                )}
                {folderFilter && folderFilter !== 'unfiled' && (
                  <button className="ghost" onClick={() => {
                    const f = folders.find(
                      (x) => String(x.id) === folderFilter)
                    if (f) setDigestFolder({ id: f.id, name: f.name })
                  }} title="A synthesized view of this whole folder over time.">
                    Folder digest
                  </button>
                )}
                <RecordMenu onStart={startRecording} />
                <button className="ghost" onClick={syncFromIphone} disabled={syncing}>
                  {syncing ? 'Syncing…' : 'Sync from iPhone'}
                </button>
              </span>
            </div>
            {(syncMsg || calMsg) && (
              <div className="toolbar-msgs">
                {calMsg && <span className="sync-msg" role="status">{calMsg}</span>}
                {syncMsg && <span className="sync-msg" role="status">{syncMsg}</span>}
              </div>
            )}
            {query.trim() && !asking && askResult == null && (
              <div className="ask-hint">
                <button className="link" onClick={ask}
                  title="Send this to the AI to answer across every meeting, whatever it looks like.">
                  Ask this instead
                </button>
              </div>
            )}

            {searchedIn && (results != null || asking) && (
              <p className="search-scope muted" role="status">
                Searching in {searchedIn.name}
                <button className="link" onClick={searchEverything}>
                  search everything
                </button>
              </p>
            )}

            {asking && (
              <p className="asking-hint muted" role="status">
                {searchedIn
                  ? `Answering from ${searchedIn.name}…`
                  : 'Answering from your meetings…'}
              </p>
            )}
            {askResult != null && (
              <section className="answer">
                <div className="section-head">
                  <h2>Answer</h2>
                  <button className="link" onClick={() => setAskResult(null)}>
                    Clear
                  </button>
                </div>
                <p>{askResult.answer}</p>
                {askResult.sources.length > 0 && (
                  <p className="sources">
                    {askResult.sources.map((s) => (
                      <button key={s.recording_id} className="link source"
                        onClick={() => setSelected(s.recording_id)}>
                        [{s.n}] {s.title || 'Untitled'}
                      </button>
                    ))}
                  </p>
                )}
              </section>
            )}

            {results != null && (
              <section>
                <div className="section-head">
                  <h2>Search results</h2>
                  <button className="link" onClick={() => { setResults(null); setQuery('') }}>
                    Clear
                  </button>
                </div>
                {results.length === 0 && <p className="muted">No matches.</p>}
                <ul className="list">
                  {results.map((r) => (
                    <li key={r.transcript_id ?? `notes-${r.recording_id}`}>
                      <button className="row" onClick={() => setSelected(r.recording_id)}>
                        <span className="row-title">
                          {r.file_path.split('/').pop()}
                          {r.source === 'notes' && (
                            <span className="topic-chip">your notes</span>
                          )}
                        </span>
                        <span
                          className="snippet"
                          dangerouslySetInnerHTML={{ __html: r.snippet }}
                        />
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <div className="filter-pills">
              <label>
                Sort
                <select value={sort} onChange={(e) => setSort(e.target.value)}>
                  {Object.entries(SORTS).map(([k, v]) => (
                    <option key={k} value={k}>{v.label}</option>
                  ))}
                </select>
              </label>
              <label>
                Speaker
                <select value={speaker} onChange={(e) => setSpeaker(e.target.value)}>
                  <option value="">All</option>
                  {allSpeakers.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </label>
              <label>
                Topic
                <select value={topic} onChange={(e) => setTopic(e.target.value)}>
                  <option value="">All</option>
                  {allTopics.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
              <label>
                From
                <input type="date" value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)} />
              </label>
              <label>
                To
                <input type="date" value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)} />
              </label>
              {filtersActive && (
                <button className="link" onClick={() => {
                  setSpeaker(''); setTopic(''); setDateFrom(''); setDateTo('')
                }}>
                  Clear filters
                </button>
              )}
              <span className="pills-count muted">
                {visible && recordings && (
                  <>
                    {filtersActive
                      ? `${visible.length} of ${recordings.length}`
                      : `${recordings.length} recordings`}
                    {processing > 0 ? ` · ${processing} processing` : ''}
                  </>
                )}
              </span>
            </div>

            <div className="list-head">
              {actionMsg && (
                <span className="sync-msg" role="status">{actionMsg}</span>
              )}
              {visible && visible.length > 0 && (
                <>
                  <button className="link" onClick={suggestFolders}
                    disabled={suggesting}
                    title="Ask the AI to suggest a folder for each unfiled recording. Nothing is filed until you accept.">
                    {suggesting ? 'Suggesting…' : 'Suggest folders'}
                  </button>
                  {pendingSuggestions > 0 && (
                    <button className="link" onClick={acceptAllSuggestions}>
                      Accept all ({pendingSuggestions})
                    </button>
                  )}
                </>
              )}
            </div>
            {selectedIds.size > 0 && visible && (
              <div className="bulk-bar">
                <button className="link" onClick={selectAllVisible}>
                  {allVisibleSelected
                    ? 'Clear selection'
                    : `Select all ${visible.length}`}
                </button>
                <span className="muted">{selectedIds.size} selected</span>
                <span className="bulk-actions">
                  <button className="ghost" onClick={moveSelected}>
                    Move to folder
                  </button>
                  <button className="ghost"
                    onClick={() => downloadExport([...selectedIds])}>
                    Export
                  </button>
                  <button className="ghost danger" onClick={deleteSelected}>
                    Delete
                  </button>
                </span>
              </div>
            )}
            <div className={wide ? 'panes' : undefined}>
            <div className={wide ? 'list-pane' : undefined}>
            {visible == null ? (
              <Spinner label="Loading recordings…" />
            ) : visible.length === 0 ? (
              <EmptyState filtered={recordings.length > 0} />
            ) : (
              <ul className="dense-list">
                {visible.map((r) => (
                  <li key={r.id}
                    className={`dense-item ${expandedId === r.id ? 'expanded' : ''} ${wide && selected === r.id ? 'selected' : ''}`}>
                    <div className="dense-row">
                      <input
                        type="checkbox"
                        className="select-box row-check"
                        checked={selectedIds.has(r.id)}
                        onChange={() => toggleSelected(r.id)}
                        aria-label={`Select ${r.title || r.file_name}`}
                      />
                      <button className="dense-main"
                        aria-expanded={wide ? undefined : expandedId === r.id}
                        aria-current={wide && selected === r.id ? 'true' : undefined}
                        onClick={() => (wide
                          ? setSelected(r.id)
                          : setExpandedId(expandedId === r.id ? null : r.id))}>
                        <span className="dense-title">{r.title || r.file_name}</span>
                        <span className="dense-chips">
                          {(r.speakers || []).filter((s) => !s.startsWith('SPEAKER_')).slice(0, 3).map((s) => (
                            <span key={s} className="speaker mini"
                              style={hueFor(s)}>{s}</span>
                          ))}
                          {r.folder && (
                            <span className="topic-chip folder-chip">{r.folder}</span>
                          )}
                        </span>
                        <span className="dense-meta">
                          <StatusChip status={r.status} />
                          {formatDate(recordingDate(r))}
                          {r.duration_seconds != null && (
                            <span className="dense-dur">{formatDuration(r.duration_seconds)}</span>
                          )}
                        </span>
                      </button>
                      {r.suggested_folder && !r.folder_id && (
                        <span className="suggestion-chip"
                          title="Suggested folder. Nothing is filed until you accept.">
                          → {r.suggested_folder}
                          <button className="tag-action confirm"
                            aria-label={`Accept folder ${r.suggested_folder}`}
                            onClick={() => acceptSuggestion(r)}>
                            ✓
                          </button>
                          <button className="tag-action reject"
                            aria-label="Dismiss folder suggestion"
                            onClick={() => dismissSuggestion(r)}>
                            ✕
                          </button>
                        </span>
                      )}
                      <RowMenu
                        onRename={() => renameRecording(r)}
                        onMove={() => moveOne(r)}
                        onRetry={(r.status === 'failed'
                                  || r.status === 'transcribing')
                          ? () => retryOne(r) : undefined}
                        onReprocess={r.status === 'done'
                          ? (scope) => reprocessOne(r, scope) : undefined}
                        onResync={isVoiceMemoName(r.file_name)
                          ? () => resyncOne(r) : undefined}
                        onDelete={() => deleteOne(r)}
                        onExport={() => downloadExport([r.id])}
                      />
                    </div>
                    {expandedId === r.id && !wide && (
                      <RowPreview recordingId={r.id}
                        onOpenFull={() => setSelected(r.id)} />
                    )}
                  </li>
                ))}
              </ul>
            )}
            </div>
            {wide && (
              <PaneHandle
                label="Resize recordings list"
                value={paneW.list}
                min={PANE_LIMITS.list[0]}
                max={PANE_LIMITS.list[1]}
                onChange={(px) => setPane('list', px)}
                onReset={() => setPane('list', PANE_DEFAULTS.list)}
              />
            )}
            {wide && (
              <div className="detail-pane">
                {selected != null ? (
                  <Detail
                    recordingId={selected}
                    onBack={() => setSelected(null)}
                    onDeleted={() => {
                      setSelected(null)
                      setActionMsg('Recording deleted')
                      refresh()
                    }}
                  />
                ) : (
                  <div className="pane-empty">
                    <svg width="40" height="40" viewBox="0 0 24 24"
                      fill="none" stroke="currentColor" strokeWidth="1.2"
                      aria-hidden="true">
                      <path strokeLinecap="round"
                        d="M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4" />
                    </svg>
                    <p>Select a recording</p>
                    <p className="muted">
                      Click a row, or move through the list with ↑ and ↓.
                    </p>
                  </div>
                )}
              </div>
            )}
            </div>
          </>
        )}
      </main>
      {recordingLayer}
    </div>
    </SpeakerColorsCtx.Provider>
  )
}
