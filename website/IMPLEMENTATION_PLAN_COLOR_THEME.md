# Implementation Plan: Environment / AQI Color Theme

## Overview

Switch the home page (and optionally site-wide) from the current purple/indigo accent palette to an environment, planet, and AQI-inspired color scheme. This document outlines palette options, file locations, and a step-by-step implementation checklist.

---

## Current State

| Token | Current Value | Usage |
|-------|---------------|-------|
| `--accent-start` | `#a855f7` (purple) | Hero gradient, CTA buttons, StatsCard, links, hover states |
| `--accent-end` | `#6366f1` (indigo) | Hero gradient, StatsCard, shadows |
| `--background` | `#ffffff` | Page background |
| `--foreground` | `#0f172a` | Text color |

**Files that reference accent colors:**
- `src/app/globals.css` — CSS variables (primary source of truth)
- `src/app/page.tsx` — Home page
- `src/app/visuals/page.tsx` — Visuals page
- `src/app/findings/page.tsx` — Findings page
- `src/app/data/page.tsx` — Data page
- `src/components/Header.tsx` — Nav links, logo
- `src/components/AnimatedBackground.tsx` — Gradient overlay, orbs, grid, particles (also has hardcoded `rgba(191, 90, 242, 0.1)` and `rgba(94, 92, 230, 0.1)`)

---

## EPA AQI Color Reference

The EPA Air Quality Index uses a standard scale:

| AQI Range | Quality | EPA Color (approx hex) |
|-----------|---------|------------------------|
| 0–50 | Good | `#00e400` (green) |
| 51–100 | Moderate | `#ffff00` (yellow) |
| 101–150 | Unhealthy for Sensitive Groups | `#ff7e00` (orange) |
| 151–200 | Unhealthy | `#ff0000` (red) |
| 201–300 | Very Unhealthy | `#8f3f97` (purple) |
| 301–500 | Hazardous | `#7e0023` (maroon) |

---

## Palette Options

### Option A: Green + Teal (Clean Air / Planet)

*Vibe: Fresh, sustainable, hopeful — "clean air for all"*

| Token | Value | Notes |
|-------|-------|-------|
| `--accent-start` | `#0d9488` (teal) | Earthy, water/sky |
| `--accent-end` | `#22c55e` (green) | Nature, growth, good AQI |

**Pros:** Strong environmental association, approachable, professional  
**Cons:** May feel generic for "eco" branding

---

### Option B: AQI-Inspired (Green → Yellow → Orange)

*Vibe: Directly references air quality scale — "good to moderate" range*

| Token | Value | Notes |
|-------|-------|-------|
| `--accent-start` | `#22c55e` (green) | Good AQI |
| `--accent-end` | `#eab308` (yellow-amber) | Moderate AQI |

**Pros:** Thematically aligned with AQI data, distinctive  
**Cons:** Yellow can be harsh at full opacity; consider softer `#facc15` or `#ca8a04`

---

### Option C: Sky Blue + Green

*Vibe: Air, sky, water — "the planet we breathe"*

| Token | Value | Notes |
|-------|-------|-------|
| `--accent-start` | `#0ea5e9` (sky blue) | Air, atmosphere |
| `--accent-end` | `#10b981` (emerald) | Earth, vegetation |

**Pros:** Calming, scientific, globally recognizable  
**Cons:** Can skew medical/healthcare if overused

---

### Option D: Green + Blue (Recommended)

*Vibe: Balanced — earth + sky, data + environment*

| Token | Value | Notes |
|-------|-------|-------|
| `--accent-start` | `#059669` (emerald) | Earth, planet, good air |
| `--accent-end` | `#0284c7` (sky blue) | Sky, air, water |

**Pros:** Readable, professional, fits environmental + data narrative; good contrast  
**Cons:** None significant

---

### Option E: Muted AQI Spectrum (Green → Orange)

*Vibe: Subtle AQI reference without loud colors*

| Token | Value | Notes |
|-------|-------|-------|
| `--accent-start` | `#16a34a` (green-600) | Softer green |
| `--accent-end` | `#ea580c` (orange-600) | Caution, moderate pollution |

**Pros:** Bold, memorable, clearly AQI-related  
**Cons:** Orange can feel aggressive; use sparingly for accents

---

## Implementation Checklist

### Phase 1: CSS Variables (Primary Change)

- [ ] **1.1** Open `src/app/globals.css`
- [ ] **1.2** Update `:root` variables:
  ```css
  :root {
    --background: #ffffff;
    --foreground: #0f172a;
    --accent-start: #059669;   /* Replace with chosen palette */
    --accent-end: #0284c7;
  }
  ```
- [ ] **1.3** If using `@theme inline`, ensure `--color-accent-start` and `--color-accent-end` map to these variables (they should already)

### Phase 2: AnimatedBackground Hardcoded Values

- [ ] **2.1** Open `src/components/AnimatedBackground.tsx`
- [ ] **2.2** Replace hardcoded grid colors (lines 25–26):
  ```tsx
  // Current (purple/indigo):
  linear-gradient(to right, rgba(191, 90, 242, 0.1) 1px, transparent 1px),
  linear-gradient(to bottom, rgba(94, 92, 230, 0.1) 1px, transparent 1px)
  ```
  With values derived from your new accent colors, e.g. for Option D:
  ```tsx
  linear-gradient(to right, rgba(5, 150, 105, 0.1) 1px, transparent 1px),
  linear-gradient(to bottom, rgba(2, 132, 199, 0.1) 1px, transparent 1px)
  ```
- [ ] **2.3** (Optional) Refactor to use CSS variables via `var(--accent-start)` — requires converting hex to RGB or using a small utility

### Phase 3: Page-by-Page Verification

- [ ] **3.1** **Home** (`page.tsx`) — Uses `accent-start`, `accent-end`; gradient text, buttons, StatsCard. Verify contrast and readability.
- [ ] **3.2** **Data** (`data/page.tsx`) — Links, code highlights. Ensure inline `accent-start` on `<code>` remains readable.
- [ ] **3.3** **Findings** (`findings/page.tsx`) — Section numbers, dividers, hover states.
- [ ] **3.4** **Visuals** (`visuals/page.tsx`) — Hero gradient, card hovers, modal accents.
- [ ] **3.5** **Header** (`Header.tsx`) — Nav hover and active states.

### Phase 4: Optional Enhancements

- [ ] **4.1** **Tailwind safelist** — If using JIT, ensure new hex values are picked up. Tailwind v4 with `@theme` should handle CSS variables automatically.
- [ ] **4.2** **Shadow colors** — `shadow-accent-end/20` may need adjustment if the new accent is lighter; consider `shadow-accent-start/30` or a neutral shadow.
- [ ] **4.3** **Dark mode** — If you add dark mode later, define `--accent-start` and `--accent-end` in a `prefers-color-scheme: dark` block or a `.dark` class.

### Phase 5: Testing

- [ ] **5.1** Run `npm run dev` and visually inspect all pages
- [ ] **5.2** Check contrast: white text on accent gradients (WCAG AA)
- [ ] **5.3** Spot-check on mobile viewport
- [ ] **5.4** Verify AnimatedBackground orbs and grid animate smoothly with new colors

---

## Recommended Palette: Option D (Green + Blue)

**Rationale:**
- **Green** (`#059669`) — Planet, nature, good air quality, environmental hope
- **Blue** (`#0284c7`) — Sky, air, water, data/science
- Readable on white backgrounds
- Distinct from typical purple/indigo tech aesthetic
- Fits the "Access to a Livable Planet" and environmental justice narrative

**Quick copy-paste for `globals.css`:**
```css
--accent-start: #059669;
--accent-end: #0284c7;
```

**AnimatedBackground grid (RGB equivalents):**
- Green: `rgba(5, 150, 105, 0.1)`
- Blue: `rgba(2, 132, 199, 0.1)`

---

## Rollback

If you need to revert, restore:
```css
--accent-start: #a855f7;
--accent-end: #6366f1;
```
And revert `AnimatedBackground.tsx` grid colors to the original purple/indigo `rgba` values.
