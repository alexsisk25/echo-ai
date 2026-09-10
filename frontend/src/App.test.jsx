import React from 'react'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import App, { contrastForHue, clampHue } from './App.jsx'

const recordings = [
  { id: 1, file_name: 'standup.m4a', duration_seconds: 65, created_at: '2026-07-07', recorded_at: '2026-07-07 09:00:00', status: 'done', transcript_id: 1, title: 'Weekly Standup', date: '2026-07-07', topics: ['shipping'], speakers: ['SPEAKER_00', 'Alexa S'] },
  { id: 2, file_name: 'notes.m4a', duration_seconds: 30, created_at: '2026-07-06', recorded_at: '2026-07-06 18:00:00', status: 'done', transcript_id: 2, title: null, date: '2026-07-06', topics: ['budget'], speakers: ['SPEAKER_00'] },
]

const detail = {
  id: 1, file_name: 'standup.m4a', duration_seconds: 65, created_at: '2026-07-07',
  status: 'done', transcript_id: 1, full_text: 'We decided to ship on Friday.',
  language: 'en', model: 'whisper',
  summary: {
    title: 'Weekly Standup', date: '2026-07-07', attendees: ['Alexa'],
    summary: 'The team agreed to ship the project on Friday.',
    decisions: ['Ship on Friday'],
    action_items: [{ owner: 'Alexa', task: 'Write the report', due_date: null, priority: 'high' }],
    risks: [], topics: ['shipping'],
  },
  segments: [
    { id: 11, start: 0, end: 3, text: 'We decided to ship on Friday.', speaker: 'SPEAKER_00', auto: false, auto_original: null, pinned: false },
    { id: 12, start: 3, end: 5, text: 'I will write the report.', speaker: 'SPEAKER_01', auto: false, auto_original: null, pinned: false },
  ],
  notes: { raw_text: '', enhanced: null, enhanced_at: null },
  reset_status: null,
}

const searchResults = [
  { transcript_id: 1, recording_id: 1, file_path: '/inbox/standup.m4a', created_at: '2026-07-07', snippet: 'ship the <mark>project</mark>' },
]

beforeEach(() => {
  global.fetch = vi.fn((url) => {
    let body = null
    if (url === '/api/recordings') body = recordings
    else if (url === '/api/folders') body = []
    else if (url === '/api/calendar/starting') body = { event: null }
    else if (url.includes('/speakers')) body = { changed: 1 }
    else if (url.startsWith('/api/recordings/1')) body = detail
    else if (url.startsWith('/api/search')) body = searchResults
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
  })
})

// Layout v2: a row expands in place; the full detail view opens from the
// preview's "Open full transcript and notes" link. This helper does both.
async function openRecording(title = 'Weekly Standup') {
  fireEvent.click(await screen.findByText(title))
  fireEvent.click(await screen.findByText(/Open full transcript and notes/))
}

describe('App', () => {
  it('lists recordings with AI titles', async () => {
    render(<App />)
    expect(await screen.findByText('Weekly Standup')).toBeInTheDocument()
    expect(screen.getByText('notes.m4a')).toBeInTheDocument()
  })

  it('shows transcript segments with speaker labels and AI notes', async () => {
    render(<App />)
    await openRecording()
    expect(await screen.findByText('We decided to ship on Friday.')).toBeInTheDocument()
    expect(screen.getByText('The team agreed to ship the project on Friday.')).toBeInTheDocument()
    expect(screen.getByText('Ship on Friday')).toBeInTheDocument()
    expect(screen.getByText('Alexa')).toBeInTheDocument()
    // Each speaker appears in the legend and on its transcript line.
    expect(screen.getAllByText('SPEAKER_00').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByLabelText('Line 1 speaker SPEAKER_00')).toBeInTheDocument()
    expect(screen.getByLabelText('Line 2 speaker SPEAKER_01')).toBeInTheDocument()
  })

  it('filters recordings by speaker', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.change(screen.getByLabelText('Speaker'), {
      target: { value: 'Alexa S' },
    })
    expect(screen.getByText('Weekly Standup')).toBeInTheDocument()
    expect(screen.queryByText('notes.m4a')).not.toBeInTheDocument()
    expect(screen.getByText('1 of 2')).toBeInTheDocument()
  })

  it('filters recordings by topic', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.change(screen.getByLabelText('Topic'), {
      target: { value: 'budget' },
    })
    expect(screen.queryByText('Weekly Standup')).not.toBeInTheDocument()
    expect(screen.getByText('notes.m4a')).toBeInTheDocument()
  })

  it('filters recordings by date range', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.change(screen.getByLabelText('From'), {
      target: { value: '2026-07-07' },
    })
    expect(screen.getByText('Weekly Standup')).toBeInTheDocument()
    expect(screen.queryByText('notes.m4a')).not.toBeInTheDocument()
  })

  it('shows and sorts by the recorded date, not the processing date', async () => {
    // Recorded in 2018, processed in 2026: the list must show 2018 and
    // put it last under newest-first.
    const rows = [
      ...recordings,
      { id: 3, file_name: 'old.m4a', duration_seconds: 10, created_at: '2026-07-07', recorded_at: '2018-03-05 12:00:00', status: 'done', transcript_id: 3, title: 'Old Memo', date: null, topics: [], speakers: [] },
    ]
    global.fetch = vi.fn((url) =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(url === '/api/recordings' ? rows : null) }),
    )
    render(<App />)
    expect(await screen.findByText('Old Memo')).toBeInTheDocument()
    const items = screen.getAllByRole('listitem').map((li) => li.textContent)
    expect(items[items.length - 1]).toContain('Old Memo')
    expect(items[items.length - 1]).toMatch(/2018/)
    expect(items[items.length - 1]).not.toMatch(/2026/)
  })

  it('sorts recordings', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.change(screen.getByLabelText('Sort'), {
      target: { value: 'shortest' },
    })
    const titles = screen.getAllByRole('listitem').map((li) => li.textContent)
    expect(titles[0]).toContain('notes.m4a')
    expect(titles[1]).toContain('Weekly Standup')
  })

  it('clicking a transcript segment seeks the audio player', async () => {
    const playMock = vi
      .spyOn(window.HTMLMediaElement.prototype, 'play')
      .mockImplementation(() => Promise.resolve())
    render(<App />)
    await openRecording()
    const audio = await screen.findByTestId('audio-player')
    fireEvent.click(screen.getByText('I will write the report.'))
    expect(audio.currentTime).toBe(3)
    expect(playMock).toHaveBeenCalled()
  })

  it('a question-shaped search fires Ask in parallel and shows the answer above results', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    const calls = []
    global.fetch.mockImplementation((url) => {
      calls.push(url)
      if (url.startsWith('/api/ask')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            answer: 'You decided to ship on Friday [1].',
            sources: [{ recording_id: 1, title: 'Weekly Standup', n: 1 }],
          }),
        })
      }
      if (url.startsWith('/api/search')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(searchResults) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.change(screen.getByLabelText('Search transcripts'), {
      target: { value: 'what did we decide' },
    })
    // One button does both: no separate Ask button exists.
    expect(screen.queryByText('Ask')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('Search'))
    // Both requests fired; answer card and keyword results both render.
    expect(await screen.findByText('You decided to ship on Friday [1].')).toBeInTheDocument()
    expect(screen.getByText('[1] Weekly Standup')).toBeInTheDocument()
    expect(await screen.findByText('Search results')).toBeInTheDocument()
    expect(calls.some((u) => u.startsWith('/api/search'))).toBe(true)
    expect(calls.some((u) => u.startsWith('/api/ask'))).toBe(true)
  })

  it('keyword-shaped queries never fire Ask; the link forces it', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    const calls = []
    global.fetch.mockImplementation((url) => {
      calls.push(url)
      if (url.startsWith('/api/ask')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({
          answer: 'Forced answer.', sources: [] }) })
      }
      if (url.startsWith('/api/search')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(searchResults) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.change(screen.getByLabelText('Search transcripts'), {
      target: { value: 'walrus budget' },
    })
    fireEvent.click(screen.getByText('Search'))
    await screen.findByText('Search results')
    // Short keyword query: search only, no LLM call.
    expect(calls.some((u) => u.startsWith('/api/ask'))).toBe(false)
    // The quiet link forces Ask when the detection guesses wrong.
    fireEvent.click(screen.getByText('Ask this instead'))
    expect(await screen.findByText('Forced answer.')).toBeInTheDocument()
    expect(calls.some((u) => u.startsWith('/api/ask'))).toBe(true)
  })

  it('question detection covers openers, question marks, and long queries', async () => {
    // Exercised through the UI: each query submits once; /api/ask
    // firing (or not) is the assertion.
    const cases = [
      ['should we hire Ana', true],
      ['zillow pricing?', true],
      ['find every mention of the equity split terms', true],
      ['walrus', false],
      ['equity split notes', false],
    ]
    for (const [q, expectAsk] of cases) {
      const calls = []
      global.fetch = vi.fn((url) => {
        calls.push(url)
        if (url.startsWith('/api/ask')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ answer: 'x', sources: [] }) })
        }
        if (url.startsWith('/api/search')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve([]) })
        }
        const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
        return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
      })
      const { unmount } = render(<App />)
      await screen.findByText('Weekly Standup')
      fireEvent.change(screen.getByLabelText('Search transcripts'), {
        target: { value: q },
      })
      fireEvent.click(screen.getByText('Search'))
      await waitFor(() =>
        expect(calls.some((u) => u.startsWith('/api/search'))).toBe(true))
      expect(calls.some((u) => u.startsWith('/api/ask')),
        `"${q}" ask=${expectAsk}`).toBe(expectAsk)
      unmount()
    }
  })

  it('every hue on the wheel clears 4.5:1 contrast in both themes', () => {
    // The whole point of the OKLCH tokens: sample the full wheel and
    // assert the minimum, so no selectable hue can go unreadable.
    let worst = { ratio: Infinity }
    for (let hue = 0; hue < 360; hue++) {
      for (const theme of ['light', 'dark']) {
        const ratio = contrastForHue(hue, theme)
        if (ratio < worst.ratio) worst = { hue, theme, ratio }
      }
    }
    expect(worst.ratio,
      `worst: hue ${worst.hue} in ${worst.theme}`).toBeGreaterThanOrEqual(4.5)
    // clampHue therefore never moves an in-range pick, and wraps others.
    expect(clampHue(178)).toBe(178)
    expect(clampHue(400)).toBe(40)
    expect(clampHue(-20)).toBe(340)
  })

  it('hue wheel: live preview while dragging, commit on release, value shown', async () => {
    const puts = []
    global.fetch = vi.fn((url, opts) => {
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = []
      else if (url === '/api/people/colors') body = {}
      else if (url.endsWith('/color') && opts?.method === 'PUT') {
        puts.push(JSON.parse(opts.body))
        body = { saved: true }
      } else if (url === '/api/people') {
        body = [{ name: 'Alexa S', meetings: 3, last_seen: '2026-07-07' }]
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getByText('People'))
    fireEvent.click(await screen.findByLabelText('Chip color for Alexa S'))
    const slider = screen.getByLabelText('Hue for Alexa S')
    // Dragging previews live on the person's real row chip, no PUT yet.
    fireEvent.change(slider, { target: { value: '200' } })
    const rowChip = screen.getAllByText('Alexa S')
      .find((el) => el.classList.contains('speaker')
        && !el.closest('.color-pop'))
    expect(rowChip.style.getPropertyValue('--hue')).toBe('200')
    expect(puts).toEqual([])
    // The resulting hue value is shown so it can be repeated deliberately.
    expect(screen.getByText('hue 200')).toBeInTheDocument()
    expect(screen.getByLabelText('Hue value for Alexa S').value).toBe('200')
    // Release commits the number.
    fireEvent.pointerUp(slider)
    await waitFor(() => expect(puts).toEqual([{ color: 200 }]))
  })

  it('speaker color: pick, propagate everywhere, duplicate note, auto reset', async () => {
    const colors = { 'Alexa S': 330, Zach: 330 }
    const puts = []
    global.fetch = vi.fn((url, opts) => {
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = []
      else if (url === '/api/people/colors') body = { ...colors }
      else if (url.endsWith('/color') && opts?.method === 'PUT') {
        puts.push([url, JSON.parse(opts.body)])
        body = { saved: true }
      } else if (url === '/api/people') {
        body = [
          { name: 'Alexa S', meetings: 3, last_seen: '2026-07-07' },
          { name: 'Zach', meetings: 1, last_seen: '2026-07-06' },
        ]
      } else if (url.startsWith('/api/recordings/1')) body = detail
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    // Propagation: the recordings-list chip uses the stored preset hue.
    await waitFor(() => {
      const chip = screen.getAllByText('Alexa S')
        .find((el) => el.classList.contains('speaker'))
      expect(chip.style.getPropertyValue('--hue')).toBe('330')
    })
    // The transcript lines use it too (open the full detail view).
    fireEvent.click(screen.getByText('Weekly Standup'))
    fireEvent.click(await screen.findByText(/Open full transcript and notes/))
    const line = await screen.findByLabelText('Line 1 speaker SPEAKER_00')
    // SPEAKER_00 has no override: automatic numeric hue.
    expect(line.style.getPropertyValue('--hue')).not.toContain('var(')

    // People tab: swatch opens the palette; live preview; duplicate note.
    fireEvent.click(screen.getByText('People'))
    const swatch = await screen.findByLabelText('Chip color for Alexa S')
    expect(swatch.style.getPropertyValue('--hue')).toBe('330')
    fireEvent.click(swatch)
    // Both people already share plum: the subtle warning names the other.
    expect(screen.getByText(/Also used by Zach/)).toBeInTheDocument()
    // Hover previews teal live in the palette preview chip and the row.
    fireEvent.mouseEnter(screen.getByLabelText('Color teal'))
    const chips = screen.getAllByText('Alexa S')
    for (const chip of chips.filter((c) => c.classList.contains('speaker'))) {
      expect(chip.style.getPropertyValue('--hue')).toBe('178')
    }
    // Committing sends the preset name, never a raw value.
    fireEvent.click(screen.getByLabelText('Color teal'))
    expect(puts).toEqual([
      ['/api/people/Alexa%20S/color', { color: 178 }],
    ])
    expect(swatch.style.getPropertyValue('--hue')).toBe('178')

    // Auto restores the automatic per-name hue and clears the override.
    fireEvent.click(swatch)
    fireEvent.click(within(document.querySelector('.color-pop'))
      .getByText(/^Auto/))
    expect(puts[1]).toEqual(['/api/people/Alexa%20S/color', { color: null }])
    expect(swatch.style.getPropertyValue('--hue')).not.toContain('var(')
  })

  it('people tab shows named speakers and opens a person page', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    global.fetch.mockImplementation((url) => {
      let body = null
      if (url === '/api/people') {
        body = [{ name: 'Zach', meetings: 3, last_seen: '2026-07-07' }]
      } else if (url === '/api/people/Zach') {
        body = {
          name: 'Zach',
          meetings: [{ recording_id: 1, created_at: '2026-07-07', title: 'Weekly Standup', date: null, summary: 's' }],
          action_items: [{ owner: 'Zach', task: 'Send the deck', due_date: null, recording_id: 1, title: 'Weekly Standup' }],
          topics: ['shipping'],
          dossier: { name: 'Zach', summary: 'Zach is a business owner.', cares_about: ['acquisitions'], commitments_made: [], follow_ups_owed: [] },
        }
      } else {
        body = recordings
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    fireEvent.click(screen.getByText('People'))
    fireEvent.click(await screen.findByText('Zach'))
    expect(await screen.findByText('Zach is a business owner.')).toBeInTheDocument()
    expect(screen.getByText('acquisitions')).toBeInTheDocument()
    expect(screen.getByText(/Send the deck/)).toBeInTheDocument()
  })

  it('people tab shows an empty state when nobody is named', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    global.fetch.mockImplementation((url) => {
      const body = url === '/api/people' ? [] : recordings
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    fireEvent.click(screen.getByText('People'))
    expect(await screen.findByText('No named speakers yet')).toBeInTheDocument()
  })

  it('commitments tab lists items and marks one done', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    const item = {
      id: 7, owner: 'Alexa', task: 'Write the report', due_date: '2026-07-10',
      priority: 'high', status: 'open', transcript_id: 1, recording_id: 1,
      meeting_title: 'Weekly Standup', meeting_date: '2026-07-07', segment_start: 3,
    }
    const posts = []
    global.fetch.mockImplementation((url, opts) => {
      if (url.startsWith('/api/commitments/') && opts?.method === 'POST') {
        posts.push([url, opts.body])
        item.status = 'done'
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 7, status: 'done' }) })
      }
      if (url === '/api/commitments') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ ...item }]) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.click(screen.getByText('Commitments'))
    expect(await screen.findByText(/Write the report/)).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText(/Mark Write the report done/))
    await waitFor(() => expect(posts).toEqual([
      ['/api/commitments/7/status', JSON.stringify({ status: 'done' })],
    ]))
  })

  it('translate button fetches and shows the translation', async () => {
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/translate') && opts?.method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            lang: 'es',
            full_text: 'Decidimos lanzar el viernes.',
            summary: null,
          }),
        })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getByText('Translate to Spanish'))
    expect(await screen.findByText('Decidimos lanzar el viernes.')).toBeInTheDocument()
    expect(screen.getByText('Show original')).toBeInTheDocument()
  })

  it('privileged toggle posts the flag', async () => {
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    const posts = []
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/privileged') && opts?.method === 'POST') {
        posts.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 1, privileged: true }) })
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ ...detail, privileged: true }),
      })
    })
    fireEvent.click(screen.getByText('Mark privileged'))
    await waitFor(() => expect(posts).toEqual([JSON.stringify({ privileged: true })]))
    expect(await screen.findByText('🔒 Privileged')).toBeInTheDocument()
  })

  it('shows the matched calendar event and unlinks it', async () => {
    const withEvent = {
      ...detail,
      calendar_event: {
        event_id: 'ev1', title: 'Board Meeting',
        start: '2025-08-17T21:00:00', end: '2025-08-17T22:00:00',
        attendees: ['Zach', 'Maria'],
      },
    }
    let current = withEvent
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url.includes('/calendar/unlink') && opts?.method === 'POST') {
        posts.push(url)
        current = { ...detail, calendar_event: null }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
      }
      if (url.startsWith('/api/recordings/1')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(current) })
      }
      const body = url === '/api/recordings' ? recordings : null
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await openRecording()
    // Board Meeting appears in both the title heading and the event
    // card; the card's strong element is what unlink removes.
    expect((await screen.findAllByText('Board Meeting')).length)
      .toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/with Zach, Maria/)).toBeInTheDocument()
    fireEvent.click(screen.getByText('Unlink'))
    await waitFor(() =>
      expect(posts).toEqual(['/api/recordings/1/calendar/unlink']))
    // After unlink the event card is gone; the heading falls back to
    // the AI summary title.
    await waitFor(() =>
      expect(screen.queryByText(/with Zach, Maria/)).not.toBeInTheDocument())
    expect(screen.getByRole('heading', { level: 2 }).textContent)
      .toContain('Weekly Standup')
  })

  it('legend rename-everywhere renames the whole speaker', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Alexa S')
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    // Two speakers, so two "Rename everywhere" buttons in the legend.
    fireEvent.click(screen.getAllByText('Rename everywhere')[0])
    await waitFor(() =>
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/transcripts/1/speakers',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            old_label: 'SPEAKER_00', new_label: 'Alexa S', force: false,
            include_pinned: false,
          }),
        }),
      ),
    )
  })

  it('a rename that changes nothing says so and offers the pinned lines', async () => {
    // The reported bug: every line of the speaker is pinned, so the
    // rename succeeds but moves nothing. It must not fail silently.
    vi.spyOn(window, 'prompt').mockReturnValue('Alexa S')
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const bodies = []
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/transcripts/1/speakers' && opts?.method === 'POST') {
        const body = JSON.parse(opts.body)
        bodies.push(body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve(
          body.include_pinned
            ? { changed: 4, enrolled_samples: 0, pinned_skipped: 0 }
            : { changed: 0, enrolled_samples: 0, pinned_skipped: 4 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getAllByText('Rename everywhere')[0])
    // It explains the no-op and asks about the pinned lines by name.
    await waitFor(() => expect(confirmSpy).toHaveBeenCalled())
    expect(confirmSpy.mock.calls[0][0]).toContain('pinned')
    // Saying yes retries including them, and reports the real outcome.
    await waitFor(() => expect(bodies.length).toBe(2))
    expect(bodies[1].include_pinned).toBe(true)
    expect(await screen.findByText(/Renamed 4 lines to "Alexa S"/)).toBeInTheDocument()
    confirmSpy.mockRestore()
  })

  it('declining the pinned-line offer leaves a visible explanation', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Alexa S')
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/transcripts/1/speakers' && opts?.method === 'POST') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(
          { changed: 0, enrolled_samples: 0, pinned_skipped: 2 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getAllByText('Rename everywhere')[0])
    expect(await screen.findByText(/2 pinned lines left as "SPEAKER_00"/)).toBeInTheDocument()
    confirmSpy.mockRestore()
  })

  it('the merge guard 409 surfaces a confirmation and forces on yes', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('SPEAKER_01')
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const bodies = []
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/transcripts/1/speakers' && opts?.method === 'POST') {
        const body = JSON.parse(opts.body)
        bodies.push(body)
        if (!body.force) {
          return Promise.resolve({ ok: false, status: 409, json: () => Promise.resolve({ detail: '"SPEAKER_01" already exists in this recording.' }) })
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ changed: 3, enrolled_samples: 0, pinned_skipped: 0 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getAllByText('Rename everywhere')[0])
    await waitFor(() => expect(confirmSpy).toHaveBeenCalled())
    expect(confirmSpy.mock.calls[0][0]).toContain('Merge them anyway?')
    await waitFor(() => expect(bodies.length).toBe(2))
    expect(bodies[1].force).toBe(true)
    expect(await screen.findByText(/Renamed 3 lines/)).toBeInTheDocument()
    confirmSpy.mockRestore()
  })

  it('a failed rename shows the server message instead of doing nothing', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Alexa S')
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/transcripts/1/speakers' && opts?.method === 'POST') {
        return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ detail: 'Database is locked' }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getAllByText('Rename everywhere')[0])
    expect(await screen.findByText('Database is locked')).toBeInTheDocument()
  })

  it('a failed per-line assign shows a message instead of failing silently', async () => {
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/segments/states')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ id: 11, speaker: 'SPEAKER_00', auto_original: null, pinned: 0 }]) })
      }
      if (url.includes('/segments/11/speaker') && opts?.method === 'POST') {
        return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({ detail: 'Segment not found' }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getByLabelText('Line 1 speaker SPEAKER_00'))
    const menu = document.querySelector('.line-menu')
    fireEvent.click(within(menu).getByText('SPEAKER_01'))
    expect(await screen.findByText('Segment not found')).toBeInTheDocument()
  })

  it('a failed undo keeps the toast and says what happened', async () => {
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/segments/states')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ id: 11, speaker: 'SPEAKER_00', auto_original: null, pinned: 0 }]) })
      }
      if (url.includes('/segments/11/speaker')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ pinned: true }) })
      }
      if (url.includes('/segments/restore')) {
        return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ detail: 'Restore failed' }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    // Assign a line to get an undo toast, then make the undo fail.
    fireEvent.click(screen.getByLabelText('Line 1 speaker SPEAKER_00'))
    fireEvent.click(within(document.querySelector('.line-menu')).getByText('SPEAKER_01'))
    fireEvent.click(await screen.findByText('Undo'))
    expect(await screen.findByText('Restore failed')).toBeInTheDocument()
    // The toast stays, so the undo can be retried.
    expect(screen.getByText('Undo')).toBeInTheDocument()
  })

  it('voices rename surfaces a plain-text server error instead of throwing', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('New Voice Name')
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/voices') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([
          { name: 'Alexa S', samples: 10, enrolled_at: '2026-07-07', recordings: 2, segments: 20, auto_segments: 5 },
        ]) })
      }
      if (url.includes('/rename') && opts?.method === 'POST') {
        // An unhandled server error returns plain text, not JSON.
        return Promise.resolve({ ok: false, status: 500, json: () => Promise.reject(new SyntaxError('Unexpected token I')) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getByText('Voices'))
    fireEvent.click(await screen.findByText('Rename'))
    expect(await screen.findByText('Rename failed (500).')).toBeInTheDocument()
  })

  it('line speaker menu assigns just that line to an existing speaker', async () => {
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    const posts = []
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/segments/states')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ id: 11, speaker: 'SPEAKER_00', auto_original: null, pinned: 0 }]) })
      }
      if (url.includes('/segments/11/speaker') && opts?.method === 'POST') {
        posts.push([url, opts.body])
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ pinned: true }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    // Open line 1's menu, pick the other existing speaker.
    fireEvent.click(screen.getByLabelText('Line 1 speaker SPEAKER_00'))
    expect(screen.getByText('This line is:')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('menuitem', { name: 'SPEAKER_01' }))
    await waitFor(() => expect(posts).toEqual([
      ['/api/transcripts/1/segments/11/speaker',
        JSON.stringify({ name: 'SPEAKER_01' })],
    ]))
    // An Undo toast appears after the action.
    expect(await screen.findByText('Undo')).toBeInTheDocument()
  })

  it('undo reverses exactly the line action', async () => {
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    const calls = []
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/segments/states')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ id: 11, speaker: 'SPEAKER_00', auto_original: null, pinned: 0 }]) })
      }
      if (url.includes('/segments/11/speaker') && opts?.method === 'POST') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ pinned: true }) })
      }
      if (url.includes('/segments/restore') && opts?.method === 'POST') {
        calls.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ restored: 1 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getByLabelText('Line 1 speaker SPEAKER_00'))
    fireEvent.click(screen.getByRole('menuitem', { name: 'SPEAKER_01' }))
    fireEvent.click(await screen.findByText('Undo'))
    await waitFor(() => expect(calls).toEqual([
      JSON.stringify({ segments: [{ id: 11, speaker: 'SPEAKER_00', auto_original: null, pinned: 0 }] }),
    ]))
  })

  it('legend group action reverts all auto-tagged lines for a speaker', async () => {
    const autoDetail = {
      ...detail,
      segments: [
        { id: 11, start: 0, end: 3, text: 'a', speaker: 'Zach', auto: true, auto_original: 'SPEAKER_00', pinned: false },
        { id: 12, start: 3, end: 5, text: 'b', speaker: 'Zach', auto: true, auto_original: 'SPEAKER_00', pinned: false },
      ],
    }
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/transcripts/1/speakers/reject' && opts?.method === 'POST') {
        posts.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ segments: 2 }) })
      }
      if (url.startsWith('/api/recordings/1')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(autoDetail) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await openRecording()
    // The legend shows the group action with its blast radius (2 auto).
    fireEvent.click(await screen.findByText('Revert all 2 auto-tagged'))
    await waitFor(() => expect(posts).toEqual([
      JSON.stringify({ label: 'Zach' }),
    ]))
  })

  it('merge rename warns and retries with force after confirmation', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Alexa S')
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    const bodies = []
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/transcripts/1/speakers' && opts?.method === 'POST') {
        bodies.push(JSON.parse(opts.body))
        if (bodies.length === 1) {
          return Promise.resolve({
            ok: false, status: 409,
            json: () => Promise.resolve({ detail: '"Alexa S" already exists in this recording. Renaming merges every line.' }),
          })
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ changed: 2 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getAllByText('Rename everywhere')[0])
    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[0].force).toBe(false)
    expect(bodies[1].force).toBe(true)
    expect(confirmSpy.mock.calls[0][0]).toContain('already exists')
  })

  it('reset speakers asks for confirmation then posts', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    const posts = []
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/speakers/reset') && opts?.method === 'POST') {
        posts.push(url)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'running' }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    fireEvent.click(screen.getByText('Reset speakers'))
    expect(confirmSpy.mock.calls[0][0]).toContain('SPEAKER_XX')
    await waitFor(() =>
      expect(posts).toEqual(['/api/recordings/1/speakers/reset']))
  })

  it('shows an empty state when there are no recordings', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({ json: () => Promise.resolve([]) }),
    )
    render(<App />)
    expect(await screen.findByText('No recordings yet')).toBeInTheDocument()
  })

  it('shows a loading indicator before recordings arrive', () => {
    global.fetch = vi.fn(() => new Promise(() => {}))
    render(<App />)
    expect(screen.getByText('Loading recordings…')).toBeInTheDocument()
  })

  it('sync button reports new recordings', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/sync' && opts?.method === 'POST') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ copied: 2 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.click(screen.getByText('Sync from iPhone'))
    expect(await screen.findByText('2 new recordings syncing…')).toBeInTheDocument()
  })

  it('sync button reports nothing new', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/sync' && opts?.method === 'POST') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ copied: 0 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.click(screen.getByText('Sync from iPhone'))
    expect(await screen.findByText('Nothing new')).toBeInTheDocument()
  })

  it('voices tab lists enrollments and deletes one with revert info', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)
    await screen.findByText('Weekly Standup')
    let voices = [{
      name: 'Zach', num_samples: 8, created_at: '2026-07-08',
      recordings: 3, segments: 42, auto_segments: 17,
    }]
    const deletes = []
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/voices/Zach' && opts?.method === 'DELETE') {
        deletes.push(url)
        voices = []
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ deleted: 'Zach', segments_reverted: 17 }) })
      }
      if (url === '/api/voices') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(voices) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.click(screen.getByText('Voices'))
    expect(await screen.findByText('Zach')).toBeInTheDocument()
    expect(screen.getByText(/3 recordings/)).toBeInTheDocument()
    expect(screen.getByText(/17 auto-tagged/)).toBeInTheDocument()
    fireEvent.click(screen.getByText('Delete'))
    await waitFor(() => expect(deletes).toEqual(['/api/voices/Zach']))
    expect(await screen.findByText('Deleted Zach (17 segments reverted)')).toBeInTheDocument()
    expect(await screen.findByText('No enrolled voices yet')).toBeInTheDocument()
  })

  it('line confirm chip confirms just that line', async () => {
    const autoDetail = {
      ...detail,
      segments: [
        { id: 11, start: 0, end: 3, text: 'a', speaker: 'Zach', auto: true, auto_original: 'SPEAKER_00', pinned: false },
        { id: 12, start: 3, end: 5, text: 'b', speaker: 'SPEAKER_01', auto: false, auto_original: null, pinned: false },
      ],
    }
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url.includes('/segments/states')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ id: 11, speaker: 'Zach', auto_original: 'SPEAKER_00', pinned: 0 }]) })
      }
      if (url === '/api/transcripts/1/segments/11/confirm' && opts?.method === 'POST') {
        posts.push(url)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ confirmed: true }) })
      }
      if (url.startsWith('/api/recordings/1')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(autoDetail) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await openRecording()
    fireEvent.click(await screen.findByLabelText('Confirm line 1'))
    await waitFor(() =>
      expect(posts).toEqual(['/api/transcripts/1/segments/11/confirm']))
  })

  it('line revert chip reverts just that line', async () => {
    const autoDetail = {
      ...detail,
      segments: [
        { id: 11, start: 0, end: 3, text: 'a', speaker: 'Zach', auto: true, auto_original: 'SPEAKER_00', pinned: false },
      ],
    }
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url.includes('/segments/states')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ id: 11, speaker: 'Zach', auto_original: 'SPEAKER_00', pinned: 0 }]) })
      }
      if (url === '/api/transcripts/1/segments/11/revert' && opts?.method === 'POST') {
        posts.push(url)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ reverted: true }) })
      }
      if (url.startsWith('/api/recordings/1')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(autoDetail) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await openRecording()
    fireEvent.click(await screen.findByLabelText('Revert line 1'))
    await waitFor(() =>
      expect(posts).toEqual(['/api/transcripts/1/segments/11/revert']))
  })

  it('multi-select lines assigns the selected lines together', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Maria')
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url.includes('/segments/states')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve([{ id: 11, speaker: 'SPEAKER_00', auto_original: null, pinned: 0 }, { id: 12, speaker: 'SPEAKER_01', auto_original: null, pinned: 0 }]) })
      }
      if (url === '/api/transcripts/1/segments/assign' && opts?.method === 'POST') {
        posts.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ assigned: 2, speaker: 'Maria' }) })
      }
      if (url.startsWith('/api/recordings/1')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    fireEvent.click(screen.getByText('Select lines'))
    fireEvent.click(screen.getByLabelText('Select line 1'))
    fireEvent.click(screen.getByLabelText('Select line 2'))
    expect(screen.getByText('2 lines selected')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Assign selected to…'))
    await waitFor(() => expect(posts).toEqual([
      JSON.stringify({ ids: [11, 12], name: 'Maria' }),
    ]))
  })

  it('renames a recording from the row menu', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('My Custom Title')
    render(<App />)
    await screen.findByText('Weekly Standup')
    const puts = []
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/recordings/1/title' && opts?.method === 'PUT') {
        puts.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 1, title: 'My Custom Title' }) })
      }
      const body = url === '/api/folders' ? [] : recordings
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    fireEvent.click(screen.getAllByLabelText('More actions')[0])
    fireEvent.click(screen.getByText('Rename'))
    await waitFor(() =>
      expect(puts).toEqual([JSON.stringify({ title: 'My Custom Title' })]))
  })

  it('moves a recording to a new folder from the row menu', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Client calls')
    render(<App />)
    await screen.findByText('Weekly Standup')
    const calls = []
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/folders' && opts?.method === 'POST') {
        calls.push(['create', opts.body])
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 7, name: 'Client calls' }) })
      }
      if (url === '/api/recordings/1/folder' && opts?.method === 'PUT') {
        calls.push(['assign', opts.body])
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 1, folder_id: 7 }) })
      }
      const body = url === '/api/folders' ? [] : recordings
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    fireEvent.click(screen.getAllByLabelText('More actions')[0])
    fireEvent.click(screen.getByText('Move to folder…'))
    await waitFor(() => expect(calls).toEqual([
      ['create', JSON.stringify({ name: 'Client calls' })],
      ['assign', JSON.stringify({ folder_id: 7 })],
    ]))
  })

  it('the folder sidebar filters the list and shows live counts', async () => {
    const rows = [
      { ...recordings[0], folder_id: 7, folder: 'Clients' },
      { ...recordings[1], folder_id: null, folder: null },
    ]
    global.fetch = vi.fn((url) => {
      let body = null
      if (url === '/api/recordings') body = rows
      else if (url === '/api/folders') body = [{ id: 7, name: 'Clients', count: 1 }]
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    // Sidebar shows All, Unfiled, and the folder, always visible.
    const sidebar = screen.getByRole('complementary')
    expect(within(sidebar).getByText('All')).toBeInTheDocument()
    expect(within(sidebar).getByText('Unfiled')).toBeInTheDocument()
    expect(within(sidebar).getByText('Clients')).toBeInTheDocument()

    fireEvent.click(within(sidebar).getByText('Clients'))
    expect(screen.getByText('Weekly Standup')).toBeInTheDocument()
    expect(screen.queryByText('notes.m4a')).not.toBeInTheDocument()

    fireEvent.click(within(sidebar).getByText('Unfiled'))
    expect(screen.queryByText('Weekly Standup')).not.toBeInTheDocument()
    expect(screen.getByText('notes.m4a')).toBeInTheDocument()
  })

  it('creates a folder from the sidebar', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('New Folder')
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/folders' && opts?.method === 'POST') {
        posts.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 9, name: 'New Folder' }) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getByText('+ New'))
    await waitFor(() =>
      expect(posts).toEqual([JSON.stringify({ name: 'New Folder' })]))
  })

  it('suggestion chip accepts with one click and never auto-files', async () => {
    const rows = [
      { ...recordings[0], suggested_folder: 'Interviews', folder_id: null, folder: null },
    ]
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/recordings/1/suggestion/accept' && opts?.method === 'POST') {
        posts.push(url)
        rows[0] = { ...rows[0], suggested_folder: null, folder_id: 9, folder: 'Interviews' }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ folder_id: 9, name: 'Interviews' }) })
      }
      let body = null
      if (url === '/api/recordings') body = [...rows]
      else if (url === '/api/folders') body = []
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    expect(screen.getByText(/→ Interviews/)).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Accept folder Interviews'))
    await waitFor(() => expect(posts).toEqual(['/api/recordings/1/suggestion/accept']))
    expect(await screen.findByText('Interviews')).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.queryByText(/→ Interviews/)).not.toBeInTheDocument())
  })

  it('row menu deletes a recording after a warning confirmation', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)
    await screen.findByText('Weekly Standup')
    const deletes = []
    global.fetch.mockImplementation((url, opts) => {
      if (opts?.method === 'DELETE') {
        deletes.push(url)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 1 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([recordings[1]]) })
    })
    fireEvent.click(screen.getAllByLabelText('More actions')[0])
    fireEvent.click(screen.getByText('Delete'))
    expect(confirmSpy.mock.calls[0][0]).toContain('immediate and permanent')
    expect(confirmSpy.mock.calls[0][0]).toContain('sync ledger')
    await waitFor(() => expect(deletes).toEqual(['/api/recordings/1']))
    await waitFor(() =>
      expect(screen.queryByText('Weekly Standup')).not.toBeInTheDocument())
  })

  it('cancelling the delete confirmation deletes nothing', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getAllByLabelText('More actions')[0])
    fireEvent.click(screen.getByText('Delete'))
    expect(global.fetch).not.toHaveBeenCalledWith(
      '/api/recordings/1', expect.objectContaining({ method: 'DELETE' }))
    expect(screen.getByText('Weekly Standup')).toBeInTheDocument()
  })

  it('row menu exports a recording as a zip download', async () => {
    const hrefs = []
    vi.spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(function () { hrefs.push(this.getAttribute('href')) })
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getAllByLabelText('More actions')[0])
    fireEvent.click(screen.getByText('Export'))
    expect(hrefs).toEqual(['/api/export?ids=1'])
  })

  it('select mode bulk deletes with one confirmation stating the count', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)
    await screen.findByText('Weekly Standup')
    const posts = []
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/recordings/bulk-delete' && opts?.method === 'POST') {
        posts.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ deleted: 2 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) })
    })
    // Checkboxes are always present; the bulk bar appears on first check.
    expect(screen.queryByText('Select')).not.toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Select Weekly Standup'))
    fireEvent.click(screen.getByText('Select all 2'))
    expect(screen.getByText('2 selected')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Delete'))
    expect(confirmSpy.mock.calls[0][0]).toContain('Delete 2 recordings?')
    await waitFor(() => expect(posts).toEqual([JSON.stringify({ ids: [1, 2] })]))
    expect(await screen.findByText('2 recordings deleted')).toBeInTheDocument()
  })

  it('checking rows exports them together; unchecking hides the bulk bar', async () => {
    const hrefs = []
    vi.spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(function () { hrefs.push(this.getAttribute('href')) })
    render(<App />)
    await screen.findByText('Weekly Standup')
    // No bulk bar until something is checked.
    expect(screen.queryByText(/selected/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Select Weekly Standup'))
    fireEvent.click(screen.getByLabelText('Select notes.m4a'))
    fireEvent.click(screen.getByText('Export'))
    expect(hrefs).toEqual(['/api/export?ids=1,2'])
    // Unchecking everything hides the bulk bar again.
    fireEvent.click(screen.getByLabelText('Select Weekly Standup'))
    fireEvent.click(screen.getByLabelText('Select notes.m4a'))
    expect(screen.queryByText(/selected/)).not.toBeInTheDocument()
    // Clicking the row itself expands it without toggling the checkbox.
    fireEvent.click(screen.getByText('Weekly Standup'))
    expect(await screen.findByText(/Open full transcript and notes/)).toBeInTheDocument()
    expect(screen.getByLabelText('Select Weekly Standup').checked).toBe(false)
  })

  it('detail page menu deletes and returns to the list', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)
    await openRecording()
    await screen.findByText('We decided to ship on Friday.')
    const deletes = []
    global.fetch.mockImplementation((url, opts) => {
      if (opts?.method === 'DELETE') {
        deletes.push(url)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ id: 1 }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([recordings[1]]) })
    })
    fireEvent.click(screen.getByLabelText('More actions'))
    fireEvent.click(screen.getByText('Delete'))
    await waitFor(() => expect(deletes).toEqual(['/api/recordings/1']))
    expect(await screen.findByText('notes.m4a')).toBeInTheDocument()
  })

  it('my notes autosave puts the text after typing pauses', async () => {
    render(<App />)
    await openRecording()
    const box = await screen.findByLabelText('My notes')
    const puts = []
    global.fetch.mockImplementation((url, opts) => {
      if (url.includes('/notes') && opts?.method === 'PUT') {
        puts.push(opts.body)
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ saved: true }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(detail) })
    })
    vi.useFakeTimers()
    fireEvent.change(box, { target: { value: 'budget??' } })
    fireEvent.change(box, { target: { value: 'budget?? ask sam' } })
    expect(puts).toEqual([])  // debounced: nothing sent yet
    expect(screen.getByText('Saving…')).toBeInTheDocument()
    await vi.advanceTimersByTimeAsync(900)
    vi.useRealTimers()
    // Only the final text is saved, once.
    expect(puts).toEqual([JSON.stringify({ text: 'budget?? ask sam' })])
    expect(await screen.findByText('Saved')).toBeInTheDocument()
  })

  it('enhance posts and renders the structured sections', async () => {
    const withNotes = {
      ...detail,
      notes: { raw_text: 'budget?? ask sam', enhanced: null, enhanced_at: null },
    }
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url.includes('/notes/enhance') && opts?.method === 'POST') {
        posts.push(url)
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            cached: false,
            enhanced: {
              sections: [
                { heading: 'Budget decision', bullets: ['Sam approved 40k for Q3'] },
              ],
            },
          }),
        })
      }
      if (opts?.method === 'PUT') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ saved: true }) })
      }
      if (url.startsWith('/api/recordings/1')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(withNotes) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    render(<App />)
    await openRecording()
    fireEvent.click(await screen.findByText('Enhance'))
    expect(await screen.findByText('Budget decision')).toBeInTheDocument()
    expect(screen.getByText('Sam approved 40k for Q3')).toBeInTheDocument()
    expect(posts).toEqual(['/api/recordings/1/notes/enhance'])
    expect(screen.getByText('Re-enhance')).toBeInTheDocument()
    // The user's own text is untouched.
    expect(screen.getByLabelText('My notes').value).toBe('budget?? ask sam')
  })

  it('search labels hits from my notes', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    global.fetch.mockImplementation((url) => {
      if (url.startsWith('/api/search')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([
            { transcript_id: null, recording_id: 1, file_path: '/inbox/standup.m4a', created_at: '2026-07-07', snippet: 'ask <mark>sam</mark>', source: 'notes' },
          ]),
        })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.change(screen.getByLabelText('Search transcripts'), {
      target: { value: 'sam' },
    })
    fireEvent.click(screen.getByText('Search'))
    expect(await screen.findByText('your notes')).toBeInTheDocument()
  })

  it('offers to record a meeting that is starting, and dismisses', async () => {
    localStorage.setItem('otterConsentAck', '1')
    let state = 'idle'
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/calendar/starting') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({
          event: { id: 'ev1', title: 'Board Sync', start: '2026-07-25T09:00:00', attendees: ['Zach'] },
        }) })
      }
      if (url === '/api/record/start' && opts?.method === 'POST') {
        posts.push(url); state = 'recording'
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state }) })
      }
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state, seconds: 0, error: null }) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    expect(await screen.findByText(/Board Sync/)).toBeInTheDocument()
    expect(screen.getByText(/with Zach/)).toBeInTheDocument()
    // "Not now" dismisses without recording.
    fireEvent.click(screen.getByText('Not now'))
    await waitFor(() =>
      expect(screen.queryByText(/is starting/)).not.toBeInTheDocument())
    expect(posts).toEqual([])
  })

  it('one-click records the starting meeting', async () => {
    localStorage.setItem('otterConsentAck', '1')
    let state = 'idle'
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/calendar/starting') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({
          event: { id: 'ev1', title: 'Board Sync', start: '2026-07-25T09:00:00', attendees: ['Zach'] },
        }) })
      }
      if (url === '/api/record/start' && opts?.method === 'POST') {
        posts.push(url); state = 'recording'
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state }) })
      }
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state, seconds: 0, error: null }) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    const banner = await screen.findByText(/Board Sync/)
    fireEvent.click(within(banner.closest('.meeting-prompt')).getByText('Record meeting'))
    await waitFor(() => expect(posts).toEqual(['/api/record/start']))
  })

  it('record meeting shows the one-time consent note then starts', async () => {
    localStorage.removeItem('otterConsentAck')
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)
    await screen.findByText('Weekly Standup')
    const posts = []
    let state = 'idle'
    global.fetch.mockImplementation((url, opts) => {
      if (url === '/api/record/start' && opts?.method === 'POST') {
        posts.push(url)
        state = 'recording'
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state }) })
      }
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state, seconds: 5, error: null }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    fireEvent.click(screen.getByText('Record meeting'))
    fireEvent.click(screen.getByText(/Online meeting/))
    await waitFor(() => expect(posts).toEqual(['/api/record/start']))
    expect(confirmSpy.mock.calls[0][0]).toContain('consent')
    expect(localStorage.getItem('otterConsentAck')).toBe('1')
    // The floating island shows elapsed time and a Stop button.
    expect(await screen.findByText('5s')).toBeInTheDocument()
    expect(screen.getByLabelText('Expand recording view')).toBeInTheDocument()

    // Second recording: no reminder again.
    confirmSpy.mockClear()
    state = 'idle'
    fireEvent.click(await screen.findByText('Stop'))
    await screen.findByText('Record meeting')
    fireEvent.click(screen.getByText('Record meeting'))
    fireEvent.click(screen.getByText(/Online meeting/))
    expect(confirmSpy).not.toHaveBeenCalled()
  })

  it('wide displays show three panes: list column plus persistent detail', async () => {
    // Pretend the window is past the 1300px breakpoint.
    window.matchMedia = (q) => ({
      matches: q.includes('1300'),
      addEventListener: () => {},
      removeEventListener: () => {},
    })
    const detail2 = { ...detail, id: 2, file_name: 'notes.m4a', transcript_id: 2, full_text: 'budget talk', summary: { ...detail.summary, title: 'Budget Chat' }, segments: [] }
    global.fetch = vi.fn((url) => {
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = []
      else if (url === '/api/calendar/starting') body = { event: null }
      else if (url.startsWith('/api/recordings/1')) body = detail
      else if (url.startsWith('/api/recordings/2')) body = detail2
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    try {
      render(<App />)
      await screen.findByText('Weekly Standup')
      // Nothing selected yet: the pane shows a tasteful empty state.
      expect(screen.getByText('Select a recording')).toBeInTheDocument()
      // Clicking a row fills the pane without leaving the list.
      fireEvent.click(screen.getByText('Weekly Standup'))
      expect(await screen.findByText('We decided to ship on Friday.')).toBeInTheDocument()
      expect(screen.getByText('notes.m4a')).toBeInTheDocument()
      // No expand-in-place in wide mode: only the pane shows the recording.
      expect(screen.queryByText(/Open full transcript and notes/)).not.toBeInTheDocument()
      // Arrow down moves the selection to the next row.
      fireEvent.keyDown(document, { key: 'ArrowDown' })
      expect(await screen.findByText('Budget Chat')).toBeInTheDocument()
      // Arrow up moves back.
      fireEvent.keyDown(document, { key: 'ArrowUp' })
      expect(await screen.findByText('We decided to ship on Friday.')).toBeInTheDocument()
    } finally {
      delete window.matchMedia
    }
  })

  it('pane seams are keyboard-resizable, clamped, reset, and remembered', async () => {
    window.matchMedia = (q) => ({
      matches: q.includes('1300'),
      addEventListener: () => {},
      removeEventListener: () => {},
    })
    localStorage.removeItem('otterPaneWidths')
    try {
      render(<App />)
      await screen.findByText('Weekly Standup')
      const handle = screen.getByRole('separator', { name: 'Resize recordings list' })
      expect(handle).toHaveAttribute('aria-valuenow', '340')
      // Arrow keys nudge, the shell variable follows, the width persists.
      fireEvent.keyDown(handle, { key: 'ArrowRight' })
      expect(JSON.parse(localStorage.getItem('otterPaneWidths')).list).toBe(356)
      const shell = document.querySelector('.app-shell')
      expect(shell.style.getPropertyValue('--list-w')).toBe('356px')
      fireEvent.keyDown(handle, { key: 'ArrowLeft' })
      expect(shell.style.getPropertyValue('--list-w')).toBe('340px')
      // Clamped at the maximum; nothing can be dragged to nothing.
      for (let i = 0; i < 30; i++) fireEvent.keyDown(handle, { key: 'ArrowRight' })
      expect(JSON.parse(localStorage.getItem('otterPaneWidths')).list).toBe(560)
      // Double-click resets the seam to its default.
      fireEvent.dblClick(handle)
      expect(JSON.parse(localStorage.getItem('otterPaneWidths')).list).toBe(340)
      // The sidebar seam exists with its own limits.
      const side = screen.getByRole('separator', { name: 'Resize sidebar' })
      expect(side).toHaveAttribute('aria-valuemin', '140')
      expect(side).toHaveAttribute('aria-valuemax', '320')
    } finally {
      delete window.matchMedia
      localStorage.removeItem('otterPaneWidths')
    }
  })

  it('speaker reprocess asks for a speaker count and sends it', async () => {
    const rows = [
      { id: 7, file_name: '20260721 122422.m4a', duration_seconds: 2520, created_at: '2026-07-21', recorded_at: '2026-07-21 12:24:22', status: 'done', transcript_id: 7, title: 'Restaurant Chat', date: null, topics: [], speakers: ['Alex Sisk'] },
    ]
    const promptSpy = vi.spyOn(window, 'prompt').mockReturnValue('2')
    // Confirm the scope dialog, then decline the denoise question.
    const confirmSpy = vi.spyOn(window, 'confirm')
      .mockReturnValueOnce(true).mockReturnValueOnce(false)
    const bodies = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/recordings/7/reprocess' && opts?.method === 'POST') {
        bodies.push(JSON.parse(opts.body))
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'running' }) })
      }
      const body = url === '/api/recordings' ? rows : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Restaurant Chat')
    fireEvent.click(screen.getByLabelText('More actions'))
    fireEvent.click(screen.getByText('Reprocess…'))
    fireEvent.click(screen.getByText('Speakers only'))
    await waitFor(() => expect(bodies).toEqual([
      { scope: 'speakers', num_speakers: 2, denoise: false },
    ]))
    expect(promptSpy.mock.calls[0][0]).toContain('How many people')
    // The denoise question is asked and explains it can hurt.
    expect(confirmSpy.mock.calls[1][0]).toContain('Clean the audio')
    promptSpy.mockRestore(); confirmSpy.mockRestore()
  })

  it('an empty speaker count lets detection decide; a bad one is refused', async () => {
    const rows = [
      { id: 7, file_name: 'a.m4a', duration_seconds: 2520, created_at: '2026-07-21', recorded_at: '2026-07-21 12:24:22', status: 'done', transcript_id: 7, title: 'Restaurant Chat', date: null, topics: [], speakers: [] },
    ]
    const bodies = []
    const mk = () => vi.fn((url, opts) => {
      if (url === '/api/recordings/7/reprocess' && opts?.method === 'POST') {
        bodies.push(JSON.parse(opts.body))
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'running' }) })
      }
      const body = url === '/api/recordings' ? rows : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    // Empty answer: no hint in the body, denoise accepted.
    let promptSpy = vi.spyOn(window, 'prompt').mockReturnValue('')
    let confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    global.fetch = mk()
    const first = render(<App />)
    await screen.findByText('Restaurant Chat')
    fireEvent.click(screen.getByLabelText('More actions'))
    fireEvent.click(screen.getByText('Reprocess…'))
    fireEvent.click(screen.getByText('Speakers only'))
    await waitFor(() => expect(bodies).toEqual([
      { scope: 'speakers', denoise: true },
    ]))
    first.unmount()

    // A nonsense count starts nothing and says why.
    promptSpy.mockReturnValue('twelve')
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})
    global.fetch = mk()
    render(<App />)
    await screen.findByText('Restaurant Chat')
    fireEvent.click(screen.getByLabelText('More actions'))
    fireEvent.click(screen.getByText('Reprocess…'))
    fireEvent.click(screen.getByText('Speakers only'))
    expect(alertSpy.mock.calls[0][0]).toContain('between 1 and 20')
    expect(bodies).toHaveLength(1)
    promptSpy.mockRestore(); confirmSpy.mockRestore(); alertSpy.mockRestore()
  })

  it('in-person recording sends the room size and remembers it', async () => {
    localStorage.setItem('otterConsentAck', '1')
    localStorage.removeItem('otterInPersonSpeakers')
    const bodies = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/record/start' && opts?.method === 'POST') {
        bodies.push(JSON.parse(opts.body))
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'recording' }) })
      }
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'idle', seconds: 0, error: null }) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getByText('Record meeting'))
    fireEvent.change(screen.getByLabelText('People in the room'), {
      target: { value: '3' },
    })
    fireEvent.click(screen.getByText(/In person/))
    await waitFor(() => expect(bodies).toEqual([
      { mode: 'in_person', num_speakers: 3 },
    ]))
    expect(localStorage.getItem('otterInPersonSpeakers')).toBe('3')
  })

  it('two voices under one name gets a quiet note with both fixes', async () => {
    const flagged = {
      ...detail, duration_seconds: 2550,
      merged_name_suspect: 'Alex Sisk', diarization_suspect: true,
    }
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url.includes('merged-name/dismiss') && opts?.method === 'POST') {
        posts.push(url)
        flagged.merged_name_suspect = null
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
      }
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = []
      else if (url === '/api/calendar/starting') body = { event: null }
      else if (url.startsWith('/api/recordings/1')) body = { ...flagged }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await openRecording()
    const note = await screen.findByText(/Two distinct voices appear under one name/)
    const box = note.closest('.suspect-msg')
    // It names the speaker and offers both routes, changing nothing.
    expect(within(box).getByText('Alex Sisk')).toBeInTheDocument()
    expect(within(box).getByText('Reset speakers')).toBeInTheDocument()
    expect(within(box).getByText('Split by hand')).toBeInTheDocument()
    expect(screen.getByText('We decided to ship on Friday.')).toBeInTheDocument()
    // The merge note wins over the generic one-speaker note.
    expect(screen.queryByText(/came out as a single speaker/)).not.toBeInTheDocument()
    // Split by hand turns on line selection without touching labels.
    fireEvent.click(within(box).getByText('Split by hand'))
    expect(await screen.findByText(/lines selected/)).toBeInTheDocument()
    fireEvent.click(within(box).getByText('Dismiss'))
    await waitFor(() => expect(posts).toHaveLength(1))
    await waitFor(() =>
      expect(screen.queryByText(/Two distinct voices/)).not.toBeInTheDocument())
  })

  it('a long one-speaker recording quietly suggests reprocessing', async () => {
    const suspect = {
      ...detail, duration_seconds: 2520, diarization_suspect: true,
    }
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url.includes('diarization-suspect/dismiss') && opts?.method === 'POST') {
        posts.push(url)
        suspect.diarization_suspect = false
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
      }
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = []
      else if (url === '/api/calendar/starting') body = { event: null }
      else if (url.startsWith('/api/recordings/1')) body = { ...suspect }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await openRecording()
    expect(await screen.findByText(/came out as a single speaker/)).toBeInTheDocument()
    // It only suggests: no labels change on its own.
    expect(screen.getByText('We decided to ship on Friday.')).toBeInTheDocument()
    fireEvent.click(within(document.querySelector('.suspect-msg')).getByText('Dismiss'))
    await waitFor(() => expect(posts).toHaveLength(1))
    await waitFor(() =>
      expect(screen.queryByText(/came out as a single speaker/)).not.toBeInTheDocument())
  })

  it('failed recording offers Retry; unreadable file points at Re-sync', async () => {
    const rows = [
      { id: 9, file_name: '20260721 122422-11F34D7B.m4a', duration_seconds: 0, created_at: '2026-07-21', recorded_at: '2026-07-21 12:24:22', status: 'failed', transcript_id: null, title: null, date: null, topics: [], speakers: [] },
    ]
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/recordings/9/retry' && opts?.method === 'POST') {
        posts.push(url)
        return Promise.resolve({ ok: false, status: 422, json: () => Promise.resolve({
          detail: 'The audio file itself is broken (the file has no moov atom, so no audio can be read). Use "Re-sync from source".',
        }) })
      }
      const body = url === '/api/recordings' ? rows : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('20260721 122422-11F34D7B.m4a')
    fireEvent.click(screen.getByLabelText('More actions'))
    // A failed recording gets Retry and Re-sync, but no Reprocess.
    expect(screen.getByText('Retry')).toBeInTheDocument()
    expect(screen.getByText('Re-sync from source')).toBeInTheDocument()
    expect(screen.queryByText('Reprocess…')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('Retry'))
    await waitFor(() => expect(posts).toEqual(['/api/recordings/9/retry']))
    expect(confirmSpy.mock.calls[0][0]).toContain('nothing to lose')
    // The unreadable-file refusal is shown plainly, naming Re-sync.
    await waitFor(() => expect(alertSpy).toHaveBeenCalled())
    expect(alertSpy.mock.calls[0][0]).toContain('Re-sync from source')
    confirmSpy.mockRestore(); alertSpy.mockRestore()
  })

  it('reprocess submenu states what survives and sends the scope', async () => {
    const rows = [
      { id: 7, file_name: '20260708 144237.m4a', duration_seconds: 60, created_at: '2026-07-08', recorded_at: '2026-07-08 14:42:37', status: 'done', transcript_id: 7, title: 'Equity Meeting', date: null, topics: [], speakers: [] },
    ]
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    // Leaving the speaker count empty lets detection decide.
    const promptSpy = vi.spyOn(window, 'prompt').mockReturnValue('')
    const bodies = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/recordings/7/reprocess' && opts?.method === 'POST') {
        bodies.push(JSON.parse(opts.body))
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'running' }) })
      }
      if (url === '/api/recordings/7/job') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ recording_status: 'done', job: { action: 'reprocess-speakers', state: 'done', error: null } }) })
      }
      const body = url === '/api/recordings' ? rows : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Equity Meeting')
    fireEvent.click(screen.getByLabelText('More actions'))
    // Done Voice Memos recording: Reprocess submenu plus Re-sync.
    expect(screen.getByText('Re-sync from source')).toBeInTheDocument()
    expect(screen.queryByText('Retry')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('Reprocess…'))
    expect(screen.getByText('Everything')).toBeInTheDocument()
    expect(screen.getByText('Notes only')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Speakers only'))
    await waitFor(() => expect(bodies).toEqual([
      { scope: 'speakers', denoise: true },
    ]))
    promptSpy.mockRestore()
    // The dialog states the preserve guarantees and the no-retranscribe rule.
    const dialog = confirmSpy.mock.calls[0][0]
    expect(dialog).toContain('WILL NOT CHANGE')
    expect(dialog.toLowerCase()).toContain('pinned lines')
    expect(dialog).toContain('NOT re-transcribed')
    confirmSpy.mockRestore()
  })

  it('record meeting offers online and in-person modes and remembers the last', async () => {
    localStorage.setItem('otterConsentAck', '1')
    localStorage.removeItem('otterRecordMode')
    localStorage.removeItem('otterInPersonSpeakers')
    let state = 'idle'
    const bodies = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/record/start' && opts?.method === 'POST') {
        bodies.push(JSON.parse(opts.body))
        state = 'recording'
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state }) })
      }
      if (url === '/api/record/stop' && opts?.method === 'POST') {
        state = 'idle'
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'idle', file: '20260726 170000-inperson.m4a' }) })
      }
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state, seconds: 3, mode: 'in_person', error: null }) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getByText('Record meeting'))
    // Both modes offered; online is the sensible default marker at first.
    expect(screen.getByText(/Online meeting/)).toBeInTheDocument()
    fireEvent.click(screen.getByText(/In person/))
    // The start request carries the chosen mode, and it is remembered.
    await waitFor(() => expect(bodies).toEqual([{ mode: 'in_person' }]))
    expect(localStorage.getItem('otterRecordMode')).toBe('in_person')
    // Next time the chooser marks in person as the last-used mode.
    state = 'idle'
    fireEvent.click(await screen.findByText('Stop'))
    await screen.findByText('Record meeting')
    fireEvent.click(screen.getByText('Record meeting'))
    expect(screen.getByText(/In person ✓/)).toBeInTheDocument()
  })

  it('in-person recording bar shows a single waveform and no meeting labels', async () => {
    localStorage.setItem('otterConsentAck', '1')
    global.fetch = vi.fn((url) => {
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'recording', seconds: 20, mode: 'in_person', error: null }) })
      }
      if (url === '/api/record/levels') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({
          recording: true, mic: 0.0, system: 0.0,
          warnings: [{ source: 'mic', message: 'Not hearing your microphone. Check it is not muted.' }],
        }) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    // The island pill flags the warning; expanding shows the focus view
    // with the mode label, one mic waveform, no You/Meeting halves.
    fireEvent.click(await screen.findByLabelText('Expand recording view'))
    expect(await screen.findByText('In person')).toBeInTheDocument()
    expect(screen.queryByText('Meeting')).not.toBeInTheDocument()
    expect(screen.queryByText('You')).not.toBeInTheDocument()
    expect(await screen.findByText(/Not hearing your microphone/)).toBeInTheDocument()
  })

  it('island expands to the focus view with labels, warning, and notes', async () => {
    localStorage.setItem('otterConsentAck', '1')
    global.fetch = vi.fn((url) => {
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'recording', seconds: 20, error: null }) })
      }
      if (url === '/api/record/levels') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({
          recording: true, mic: 0.3, system: 0.0,
          warnings: [{ source: 'system', message: 'Not hearing any meeting audio. Make sure the call is playing out loud.' }],
        }) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    // Collapsed: a pill with elapsed time and a warning glyph. Expanded:
    // the mirrored waveform labeled You/Meeting, the mode label, the
    // full warning text, and the My notes box.
    fireEvent.click(await screen.findByLabelText('Expand recording view'))
    expect(await screen.findByText('You')).toBeInTheDocument()
    expect(screen.getByText('Meeting')).toBeInTheDocument()
    expect(screen.getByText('Online meeting')).toBeInTheDocument()
    expect(await screen.findByText(/Not hearing any meeting audio/)).toBeInTheDocument()
    expect(screen.getByText('My notes')).toBeInTheDocument()
    // Escape collapses back to the pill without stopping anything.
    fireEvent.keyDown(document, { key: 'Escape' })
    await waitFor(() =>
      expect(screen.queryByText('My notes')).not.toBeInTheDocument())
    expect(screen.getByLabelText('Expand recording view')).toBeInTheDocument()
  })

  it('notes typed in the island attach to the recording when it lands', async () => {
    localStorage.setItem('otterConsentAck', '1')
    localStorage.removeItem('otterIslandNotes')
    localStorage.removeItem('otterPendingNotes')
    let state = 'recording'
    let rows = []
    const notePuts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/record/stop' && opts?.method === 'POST') {
        state = 'idle'
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: 'idle', file: 'landed.m4a' }) })
      }
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state, seconds: 9, error: null }) })
      }
      if (url === '/api/recordings/42/notes' && opts?.method === 'PUT') {
        notePuts.push(JSON.parse(opts.body))
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ saved: true }) })
      }
      const body = url === '/api/recordings' ? rows : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    fireEvent.click(await screen.findByLabelText('Expand recording view'))
    fireEvent.change(screen.getByPlaceholderText(/Jot as you listen/), {
      target: { value: 'ask about the Q3 numbers' },
    })
    // Stop while the recording has not landed yet: the notes wait.
    rows = [{ id: 42, file_name: 'landed.m4a', duration_seconds: 9, created_at: '2026-07-26', recorded_at: '2026-07-26 10:00:00', status: 'done', transcript_id: 9, title: 'Landed', date: null, topics: [], speakers: [] }]
    fireEvent.click(screen.getByText('Stop'))
    // The refresh returns the landed recording; the notes attach to it.
    await waitFor(() => expect(notePuts).toEqual([{ text: 'ask about the Q3 numbers' }]))
    expect(localStorage.getItem('otterPendingNotes')).toBe(null)
    expect(await screen.findByText(/notes were attached/)).toBeInTheDocument()
  })

  it('stop recording saves the file and reports it', async () => {
    localStorage.setItem('otterConsentAck', '1')
    let state = 'recording'
    const posts = []
    global.fetch = vi.fn((url, opts) => {
      if (url === '/api/record/stop' && opts?.method === 'POST') {
        posts.push(url)
        state = 'idle'
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ state: 'idle', file: '20260717 210000-meeting.m4a' }),
        })
      }
      if (url === '/api/record/status') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state, seconds: 65, error: null }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
    render(<App />)
    fireEvent.click(await screen.findByText('Stop'))
    await waitFor(() => expect(posts).toEqual(['/api/record/stop']))
    expect(await screen.findByText(/Saved 20260717 210000-meeting.m4a/)).toBeInTheDocument()
    expect(await screen.findByText('Record meeting')).toBeInTheDocument()
  })

  it('activity tab lists provenance events and filters by type', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    const evs = [
      { id: 3, created_at: '2026-07-26 12:00:00', event_type: 'deleted', stem: '20260126 110850-X', context: { title: 'Old Sync', how: 'maintenance', context: 'qta incident cleanup' } },
      { id: 2, created_at: '2026-07-24 11:07:00', event_type: 'imported', stem: '20260724 110715-Y', context: { original_filename: '20260724 110715-Y.qta', source_format: '.qta' } },
    ]
    global.fetch.mockImplementation((url) => {
      if (url.startsWith('/api/activity')) {
        const filtered = url.includes('type=deleted')
          ? evs.filter((e) => e.event_type === 'deleted') : evs
        return Promise.resolve({ ok: true, json: () => Promise.resolve(filtered) })
      }
      const body = url === '/api/recordings' ? recordings : (url === '/api/folders' ? [] : null)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    fireEvent.click(screen.getByText('Activity'))
    expect(await screen.findByText(/qta incident cleanup/)).toBeInTheDocument()
    expect(screen.getByText(/20260724 110715-Y.qta/)).toBeInTheDocument()
    // Filter to deleted only.
    fireEvent.change(screen.getByLabelText('Type'), { target: { value: 'deleted' } })
    await waitFor(() =>
      expect(screen.queryByText(/20260724 110715-Y.qta/)).not.toBeInTheDocument())
    expect(screen.getByText(/qta incident cleanup/)).toBeInTheDocument()
  })

  it('three-state theme control pins light or dark and Auto follows the system', async () => {
    localStorage.removeItem('echoTheme')
    delete document.documentElement.dataset.theme
    render(<App />)
    await screen.findByText('Weekly Standup')
    // Auto is the default and marked active.
    const group = screen.getByRole('group', { name: 'Theme' })
    expect(within(group).getByText('Auto')).toHaveAttribute('aria-pressed', 'true')
    expect(document.documentElement.dataset.theme).toBeUndefined()
    // Dark pins dark and persists.
    fireEvent.click(within(group).getByText('Dark'))
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem('echoTheme')).toBe('dark')
    expect(within(group).getByText('Dark')).toHaveAttribute('aria-pressed', 'true')
    // Light pins light.
    fireEvent.click(within(group).getByText('Light'))
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(localStorage.getItem('echoTheme')).toBe('light')
    // Auto releases the pin back to the system and persists the choice.
    fireEvent.click(within(group).getByText('Auto'))
    expect(document.documentElement.dataset.theme).toBeUndefined()
    expect(localStorage.getItem('echoTheme')).toBe('auto')
  })

  it('shows the Echo AI wordmark in the header', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    expect(screen.getByText('Echo')).toBeInTheDocument()
    expect(screen.getByText('ai')).toBeInTheDocument()
  })

  it('searches by keyword and shows snippets', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.change(screen.getByLabelText('Search transcripts'), {
      target: { value: 'project' },
    })
    fireEvent.click(screen.getByText('Search'))
    expect(await screen.findByText('Search results')).toBeInTheDocument()
    expect(screen.getByText('standup.m4a')).toBeInTheDocument()
    await waitFor(() =>
      expect(global.fetch).toHaveBeenCalledWith('/api/search?q=project'),
    )
  })

  // ---------- folder digests ----------

  const folderRows = [{ id: 7, name: 'Clients', count: 2, privileged: false }]

  const digestBody = {
    overview: 'Pricing came up in January and kept coming back.',
    themes: [{ theme: 'pricing', first_appeared: '2026-01-05',
      framing_then: 'a cost problem', framing_now: 'a packaging problem',
      how_it_changed: 'reframed in February' }],
    shifts: [{ date: '2026-02-02', shift: 'moved to annual billing', note: '' }],
    open_threads: [{ thread: 'who owns the renewal', raised_on: '2026-01-05',
      last_mentioned: '2026-02-02', note: 'never named' }],
    commitments: [{ owner: 'Sam', commitment: 'send the forecast',
      made_on: '2026-01-05', status: 'open', note: '' }],
    quotes: [{ quote: 'we should revisit pricing', speaker: 'Sam',
      recording_id: 1, timestamp: '00:00:03', start_seconds: 3, why: 'sets it up' }],
  }

  const digestState = (over = {}, stored = {}) => ({
    folder: { id: 7, name: 'Clients', privileged: false },
    privileged: false,
    recordings_in_folder: 2,
    digest: {
      folder_id: 7, digest: digestBody, covered_ids: [1, 2],
      recordings_count: 2, stale: false, model: 'cloud',
      updated_at: '2026-08-18 14:00:00', new_since: 0, ...stored,
    },
    ...over,
  })

  const digestFetch = (state, calls = []) => vi.fn((url, opts) => {
    let body = null
    if (url === '/api/recordings') body = recordings
    else if (url === '/api/folders') body = folderRows
    else if (url.startsWith('/api/folders/7/digest')) {
      calls.push({ url, method: opts?.method || 'GET', body: opts?.body })
      body = state
    } else if (url.startsWith('/api/folders/7/privileged')) {
      calls.push({ url, method: opts?.method, body: opts?.body })
      body = { folder_id: 7, privileged: true, recordings_marked: 2 }
    } else if (url.startsWith('/api/ask')) {
      calls.push({ url, method: 'GET' })
      body = { answer: 'They kept revisiting pricing.',
        sources: [{ recording_id: 1, title: 'January', n: 1 }] }
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
  })

  async function openDigest(state, calls) {
    global.fetch = digestFetch(state, calls)
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.click(screen.getByLabelText('Digest of Clients'))
  }

  it('shows a folder digest with what it was generated from', async () => {
    const calls = []
    await openDigest(digestState(), calls)

    expect(await screen.findByText('Clients digest')).toBeInTheDocument()
    expect(screen.getByText(/Generated from 2 recordings, last updated/))
      .toBeInTheDocument()
    expect(screen.getByText(digestBody.overview)).toBeInTheDocument()
    // Trajectory, not aggregation: themes carry when and how they changed.
    expect(screen.getByText('pricing')).toBeInTheDocument()
    expect(screen.getByText(/first appeared/)).toBeInTheDocument()
    expect(screen.getByText('a cost problem')).toBeInTheDocument()
    expect(screen.getByText('a packaging problem')).toBeInTheDocument()
    expect(screen.getByText('moved to annual billing')).toBeInTheDocument()
    expect(screen.getByText(/who owns the renewal/)).toBeInTheDocument()
    expect(screen.getByText(/send the forecast/)).toBeInTheDocument()
    // Every quote links back to its recording and timestamp.
    expect(screen.getByText(/we should revisit pricing/)).toBeInTheDocument()
    expect(screen.getByText(/Sam · 00:00:03/)).toBeInTheDocument()
    expect(calls[0].method).toBe('GET')
  })

  it('generates a digest for a folder that has none yet', async () => {
    const calls = []
    await openDigest(digestState({ digest: null }), calls)

    expect(await screen.findByText(/No digest yet\. 2 recordings/))
      .toBeInTheDocument()
    fireEvent.click(screen.getByText('Generate digest'))
    await waitFor(() => expect(calls.some(
      (c) => c.method === 'POST' && c.body === JSON.stringify({ force: false }),
    )).toBe(true))
  })

  it('flags recordings added since the digest and folds them in', async () => {
    const calls = []
    await openDigest(digestState({}, { stale: true, new_since: 1 }), calls)

    expect(await screen.findByText(/1 recording added since this digest/))
      .toBeInTheDocument()
    fireEvent.click(screen.getByText('Update'))
    await waitFor(() => expect(calls.some(
      (c) => c.method === 'POST' && c.body === JSON.stringify({ force: false }),
    )).toBe(true))
    // Rebuilding is the separate, explicit action.
    fireEvent.click(screen.getByText('Rebuild from scratch'))
    await waitFor(() => expect(calls.some(
      (c) => c.method === 'POST' && c.body === JSON.stringify({ force: true }),
    )).toBe(true))
  })

  it('asks a question scoped to the folder', async () => {
    const calls = []
    await openDigest(digestState(), calls)
    await screen.findByText('Clients digest')

    fireEvent.change(screen.getByLabelText('Ask about Clients'), {
      target: { value: 'what happened with pricing?' },
    })
    fireEvent.click(screen.getByText('Ask this folder'))
    expect(await screen.findByText('They kept revisiting pricing.'))
      .toBeInTheDocument()
    const ask = calls.find((c) => c.url.startsWith('/api/ask'))
    expect(ask.url).toContain('folder_id=7')
    expect(ask.url).toContain('q=what%20happened%20with%20pricing%3F')
  })

  it('marks a whole folder privileged after saying what changes', async () => {
    const calls = []
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    await openDigest(digestState(), calls)
    await screen.findByText('Clients digest')

    fireEvent.click(screen.getByLabelText(/Keep this folder local/))
    await waitFor(() => expect(calls.some(
      (c) => c.method === 'PUT'
        && c.body === JSON.stringify({ privileged: true }),
    )).toBe(true))
    expect(confirm.mock.calls[0][0]).toMatch(/every recording in it is marked/)
    confirm.mockRestore()
  })

  it('says when a digest is local only', async () => {
    await openDigest(digestState({
      privileged: true,
      folder: { id: 7, name: 'Clients', privileged: true },
    }))
    expect(await screen.findByText(/local model only/)).toBeInTheDocument()
  })
})

// Due dates were stored and printed but never compared to today, so an
// overdue promise read exactly like one due next year.
describe('commitment due states', () => {
  const rows = [
    { id: 1, owner: 'Sam', task: 'Send the deck', due_date: '2026-09-01',
      priority: null, status: 'open', transcript_id: 1, recording_id: 1,
      meeting_title: 'Board', meeting_date: '2026-08-30', segment_start: 0,
      due_state: 'overdue', days_until: -8, mine: false },
    { id: 2, owner: 'Alex Sisk', task: 'Book the venue', due_date: '2026-09-11',
      priority: null, status: 'open', transcript_id: 1, recording_id: 1,
      meeting_title: 'Board', meeting_date: '2026-08-30', segment_start: 0,
      due_state: 'due-soon', days_until: 2, mine: true },
    { id: 3, owner: 'Sam', task: 'Renew the lease', due_date: '2027-01-01',
      priority: null, status: 'open', transcript_id: 1, recording_id: 1,
      meeting_title: 'Board', meeting_date: '2026-08-30', segment_start: 0,
      due_state: 'later', days_until: 114, mine: false },
  ]

  function mockCommitments(seen) {
    global.fetch.mockImplementation((url, opts) => {
      if (url.startsWith('/api/commitments') && opts?.method !== 'POST') {
        seen.push(url)
        const u = new URL(url, 'http://x')
        let out = rows
        const due = u.searchParams.get('due')
        if (due) out = out.filter((r) => r.due_state === due)
        const mine = u.searchParams.get('mine')
        if (mine === 'true') out = out.filter((r) => r.mine)
        if (mine === 'false') out = out.filter((r) => !r.mine)
        return Promise.resolve({ ok: true, json: () => Promise.resolve(out) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(recordings) })
    })
  }

  it('shows overdue and due-soon chips and a header count', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    mockCommitments([])
    fireEvent.click(screen.getByText('Commitments'))

    expect(await screen.findByText(/Overdue · 8 days/)).toBeInTheDocument()
    expect(screen.getByText(/Due soon · 2 days/)).toBeInTheDocument()
    expect(screen.getByText('1 overdue')).toBeInTheDocument()
    expect(screen.getByText('1 due soon')).toBeInTheDocument()
    // Nothing urgent about a promise due in four months.
    expect(screen.queryByText(/Overdue · 114/)).not.toBeInTheDocument()
  })

  it('asks the server for overdue only, and for mine only', async () => {
    render(<App />)
    await screen.findByText('Weekly Standup')
    const seen = []
    mockCommitments(seen)
    fireEvent.click(screen.getByText('Commitments'))
    await screen.findByText(/Send the deck/)

    fireEvent.change(screen.getByLabelText('When'),
      { target: { value: 'overdue' } })
    await waitFor(() => expect(
      seen.some((u) => u.includes('due=overdue'))).toBe(true))
    await waitFor(() => expect(
      screen.queryByText(/Book the venue/)).not.toBeInTheDocument())

    fireEvent.change(screen.getByLabelText('When'), { target: { value: '' } })
    fireEvent.change(screen.getByLabelText('Whose'),
      { target: { value: 'mine' } })
    await waitFor(() => expect(
      seen.some((u) => u.includes('mine=true'))).toBe(true))
    expect(await screen.findByText(/Book the venue/)).toBeInTheDocument()
  })
})

// The top search bar stayed global while the list was filtered to a
// folder, so hits from other folders looked like a broken filter.
describe('folder-scoped search', () => {
  const folders = [{ id: 7, name: 'Therapy', count: 1, privileged: false }]

  it('scopes search and ask to the filtered folder, and can widen', async () => {
    const seen = []
    global.fetch = vi.fn((url) => {
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = folders
      else if (url === '/api/calendar/starting') body = { event: null }
      else if (url.startsWith('/api/search')) { seen.push(url); body = searchResults }
      else if (url.startsWith('/api/ask')) { seen.push(url); body = { answer: 'Yes.', sources: [] } }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')

    fireEvent.click(screen.getByRole('button', { name: /^Therapy/ }))
    const box = await screen.findByLabelText('Search Therapy')
    fireEvent.change(box, { target: { value: 'what did we decide about rent?' } })
    fireEvent.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => expect(
      seen.some((u) => u.startsWith('/api/search') && u.includes('folder_id=7'))
    ).toBe(true))
    await waitFor(() => expect(
      seen.some((u) => u.startsWith('/api/ask') && u.includes('folder_id=7'))
    ).toBe(true))
    expect(await screen.findByText(/Searching in Therapy/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'search everything' }))
    await waitFor(() => expect(
      seen.some((u) => u.startsWith('/api/search') && !u.includes('folder_id'))
    ).toBe(true))
    await waitFor(() => expect(
      screen.queryByText(/Searching in Therapy/)).not.toBeInTheDocument())
  })

  it('leaves search global when no folder is filtered', async () => {
    const seen = []
    global.fetch = vi.fn((url) => {
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = folders
      else if (url === '/api/calendar/starting') body = { event: null }
      else if (url.startsWith('/api/search')) { seen.push(url); body = searchResults }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
    render(<App />)
    await screen.findByText('Weekly Standup')
    fireEvent.change(screen.getByLabelText('Search transcripts'),
      { target: { value: 'rent' } })
    fireEvent.click(screen.getByRole('button', { name: 'Search' }))
    await waitFor(() => expect(seen.length).toBe(1))
    expect(seen[0]).not.toContain('folder_id')
    expect(screen.queryByText(/Searching in/)).not.toBeInTheDocument()
  })
})

// PLAN Phase 3: "fall back to asking me when ambiguous".
describe('which meeting was this', () => {
  const ambiguous = {
    ...detail,
    calendar_event: null,
    calendar_candidates: [
      { id: 'a', title: 'Board Meeting', start: '2026-09-09T10:00:00',
        end: '2026-09-09T11:00:00', attendees: ['Sam'] },
      { id: 'b', title: 'Investor Call', start: '2026-09-09T10:05:00',
        end: '2026-09-09T11:05:00', attendees: ['Dana'] },
    ],
  }

  function mockDetail(posts) {
    global.fetch = vi.fn((url, opts) => {
      let body = null
      if (url === '/api/recordings') body = recordings
      else if (url === '/api/folders') body = []
      else if (url === '/api/calendar/starting') body = { event: null }
      else if (url.includes('/calendar/choose')) {
        posts.push(JSON.parse(opts.body))
        body = { calendar_event: null, calendar_candidates: [] }
      } else if (url.startsWith('/api/recordings/1')) {
        body = posts.length ? { ...detail, calendar_candidates: [] } : ambiguous
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
  }

  it('offers both events and sends the one picked', async () => {
    const posts = []
    mockDetail(posts)
    render(<App />)
    await openRecording()
    expect(await screen.findByText('Which meeting was this?')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Investor Call/ }))
    await waitFor(() => expect(posts).toEqual([{ event_id: 'b' }]))
    await waitFor(() => expect(
      screen.queryByText('Which meeting was this?')).not.toBeInTheDocument())
  })

  it('sends null for neither', async () => {
    const posts = []
    mockDetail(posts)
    render(<App />)
    await openRecording()
    await screen.findByText('Which meeting was this?')
    fireEvent.click(screen.getByRole('button', { name: 'Neither' }))
    await waitFor(() => expect(posts).toEqual([{ event_id: null }]))
  })
})
