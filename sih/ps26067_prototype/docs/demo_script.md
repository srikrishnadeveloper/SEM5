# SIH26067 — 3-Minute Demo Script

**Total duration:** ~3 minutes  
**Tone:** clear, confident, science + engineering  
**Setup:** browser open to `http://localhost:8765`, full-screen, 3D globe visible

---

## 0:00–0:30 | Introduction & Hook

**What to say:**

> "Good morning/afternoon judges. We are Team [Name] from SSN College of Engineering, and we are here to present SIH26067 — a browser-native 3D ocean visualization platform that brings together numerical model outputs and in-situ observations in one interactive view."

**On screen:** Title slide

**Speaker notes:**

- Make eye contact and pause after the hook.
- Mention the problem statement in one line: integrating models and observations.
- Set the expectation: "Let us show you the problem, the demo, and the impact in the next three minutes."

---

## 0:30–1:00 | Problem & Gaps

**What to say:**

> "Today ocean scientists rely on separate tools — 2D maps, spreadsheets and desktop software — to look at model forecasts and instrument observations. There is no single, web-native 3D view that lets you see both at once. The key gaps are: no real 3D globe with depth slicing, no time animation of forecast steps, no current vectors, and no way to click an Argo float and see its depth profile. This makes analysis slow and limits situational awareness."

**On screen:** Problem Statement & Gaps slide

**Speaker notes:**

- Use hand motion to show "separate tools" and the missing unified view.
- Emphasize the words "web-native" and "3D".
- End with: "That is exactly what our platform solves."

---

## 1:00–2:30 | Demo Walkthrough

### 1:00–1:15 | 3D Globe & Variable Selection

**What to say:**

> "This is the Cesium.js 3D globe, free of any paid tokens, using an OpenStreetMap basemap. You can pan, zoom and rotate. From the sidebar I can switch ocean variables — temperature, salinity, chlorophyll and current speed — each with its own scientific color scale."

**Actions:**

- Slightly rotate the globe.
- Open the variable dropdown and switch to **Temperature**.

**Speaker notes:**

- Move the mouse slowly so judges can follow.
- Highlight "no paid API keys" since the OSM basemap is free.
- Point to the color scale on the side.

---

### 1:15–1:35 | Depth Slice

**What to say:**

> "The depth slider moves a horizontal slice through the water column from the surface down to 500 metres. Watch how the temperature field changes — warm surface water cools as we go deeper. The slice is mapped to real latitude and longitude."

**Actions:**

- Drag the depth slider from **0 m** to **100 m**.
- Optionally switch to **Salinity** to show a different pattern.

**Speaker notes:**

- Mention the discrete depth levels: 0, 50, 100, 200 and 500 m.
- Point out the color scale and the current depth label.

---

### 1:35–1:55 | Current Vectors

**What to say:**

> "I can overlay current vectors. The arrows show U and V flow components, and their color shows speed. The pattern is inspired by seasonal monsoon circulation, so you can see the broad flow direction over the Arabian Sea and Bay of Bengal."

**Actions:**

- Toggle the **Current Vectors** layer.
- Zoom to the Indian Ocean region.

**Speaker notes:**

- Mention U/V components and speed-based coloring.
- Keep this section smooth; do not over-explain the physics.

---

### 1:55–2:15 | Time Animation

**What to say:**

> "The time slider or play button steps through six synthetic forecast and observation time steps. You can watch the chlorophyll bloom or the current field evolve. This is exactly what the problem statement asks for — time-step animation of model outputs."

**Actions:**

- Press **Play** and let it run through two or three steps.
- Pause on a step where the change is visible.

**Speaker notes:**

- Pre-generated tiles keep the animation smooth.
- Mention the play/pause button and the current time label.

---

### 2:15–2:30 | Instruments & Profiles

**What to say:**

> "Finally, the instrument layer. Argo floats, gliders, CTDs and BGC-Argo markers are plotted at true lat/lon/depth. When I click an Argo float, we get a depth-versus-variable profile chart. I can also click anywhere on the data layer to probe the exact model value at that lat/lon/depth — model output and in-situ observation co-visualized."

**Actions:**

- Click an **Argo float** marker on the globe or click the data layer to probe any lat/lon value.
- Scroll through the profile chart.

**Speaker notes:**

- Show the depth profile clearly and read one sample value.
- Mention that instruments are clickable and informative.

---

## 2:30–3:00 | Impact & Conclusion

**What to say:**

> "Our platform directly addresses the SIH26067 brief. It is open-source, offline-capable and built on a modular data adapter, so replacing the synthetic source with real NetCDF or OPeNDAP data is a one-file change. The impact is real: better fisheries planning, climate monitoring, marine operations and disaster response. Thank you, and we welcome your questions."

**On screen:** Impact, Feasibility & Future slide

**Speaker notes:**

- End strong and pause for a second.
- Gesture toward the screen when you say "impact".
- If ahead of schedule, slow down and emphasize the modular data adapter.

---

## Quick Tips

- Keep the browser full-screen and the mouse cursor visible at all times.
- Do not click faster than you speak.
- Have a fallback static screenshot ready in case the live demo fails.
- Test the demo once on the exact machine and screen before the pitch.
