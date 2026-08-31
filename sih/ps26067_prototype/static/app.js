const API = '';

let viewer;
let dataLayer = null;
let instrumentEntities = [];
let vectorEntities = [];
let profileChart = null;

let variables = [];
let depths = [];
let times = [];
let bounds = null;
let selectedInstrument = null;

let playInterval = null;
let currentSlice = null;

const GRADIENT_PRESETS = {
    temperature: 'linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #fee090, #fdae61, #f46d43, #a50026)',
    salinity: 'linear-gradient(to right, #f7fbff, #c6dbef, #6baed6, #2171b5, #08306b)',
    chlorophyll: 'linear-gradient(to right, #ffffcc, #c7e9b4, #7fcdbb, #1d91c0, #0c2c84)',
    current_speed: 'linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #f9c631, #f0f921)',
};

function showLoading(text = 'Loading...') {
    const el = document.getElementById('loading');
    if (el) {
        el.querySelector('p').textContent = text;
        el.classList.add('visible');
    }
}

function hideLoading() {
    const el = document.getElementById('loading');
    if (el) el.classList.remove('visible');
}

function showError(msg) {
    const banner = document.createElement('div');
    banner.className = 'error-banner';
    banner.textContent = msg;
    document.body.appendChild(banner);
    setTimeout(() => banner.remove(), 6000);
}

async function init() {
    try {
        const cfg = await (await fetch(`${API}/api/variables`)).json();
        variables = cfg.variables;
        depths = cfg.depths;
        times = cfg.times;
        bounds = cfg.bounds;

        setupControls();
        setupCesium();
        await loadDataLayer();
        await loadInstruments();
        await loadVectors();
        setupEvents();
        hideLoading();
    } catch (err) {
        console.error(err);
        hideLoading();
        showError('Failed to initialize 3D globe. Check console.');
    }
}

function setupControls() {
    const varSel = document.getElementById('variable');
    varSel.innerHTML = '';
    variables.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.id;
        opt.textContent = `${v.name} (${v.unit})`;
        varSel.appendChild(opt);
    });

    const depthSlider = document.getElementById('depth');
    depthSlider.max = depths.length - 1;
    depthSlider.value = 0;
    document.getElementById('depthVal').textContent = `${depths[0]} m`;

    const ticks = document.getElementById('depthTicks');
    ticks.innerHTML = '';
    depths.forEach(d => {
        const span = document.createElement('span');
        span.textContent = d;
        ticks.appendChild(span);
    });

    const timeSlider = document.getElementById('time');
    timeSlider.max = times.length - 1;
    timeSlider.value = 0;
    document.getElementById('timeVal').textContent = `T${times[0]}`;
}

function setupCesium() {
    // No Cesium Ion token is needed because the globe uses the free OSM basemap
    // and local SingleTileImageryProvider overlays.

    const osmProvider = new Cesium.UrlTemplateImageryProvider({
        url: 'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
        maximumLevel: 19,
    });

    viewer = new Cesium.Viewer('cesiumContainer', {
        imageryProvider: osmProvider,
        baseLayerPicker: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        timeline: false,
        navigationHelpButton: false,
        fullscreenButton: false,
        animation: false,
        vrButton: false,
        infoBox: false,
        selectionIndicator: false,
        shouldAnimate: false,
    });

    if (viewer.cesiumWidget && viewer.cesiumWidget.creditContainer) {
        viewer.cesiumWidget.creditContainer.style.display = 'none';
    }

    viewer.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(bounds.lon_min, bounds.lat_min, bounds.lon_max, bounds.lat_max),
        duration: 0,
    });

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction(handleClick, Cesium.ScreenSpaceEventType.LEFT_CLICK);
}

async function loadDataLayer() {
    const variable = document.getElementById('variable').value;
    const depth = depths[document.getElementById('depth').value];
    const time = times[document.getElementById('time').value];

    hideProbe();
    showLoading(`Loading ${variables.find(v => v.id === variable).name}...`);

    try {
        const meta = await (await fetch(`${API}/api/slice?variable=${variable}&depth=${depth}&time=${time}`)).json();
        currentSlice = meta;
        updateLegend(variable, meta);

        const imageUrl = `${API}/api/image?variable=${variable}&depth=${depth}&time=${time}&_=${Date.now()}`;
        const rectangle = Cesium.Rectangle.fromDegrees(bounds.lon_min, bounds.lat_min, bounds.lon_max, bounds.lat_max);

        const provider = new Cesium.SingleTileImageryProvider({
            url: imageUrl,
            rectangle: rectangle,
        });

        if (dataLayer) {
            viewer.imageryLayers.remove(dataLayer, true);
        }

        dataLayer = viewer.imageryLayers.addImageryProvider(provider);
        dataLayer.alpha = 0.85;
        dataLayer.brightness = 1.0;
        dataLayer.contrast = 1.0;
    } catch (err) {
        console.error(err);
        showError('Failed to load data layer.');
    } finally {
        hideLoading();
    }
}

function updateLegend(variable, meta) {
    const metaVar = variables.find(v => v.id === variable);
    document.getElementById('legendTitle').textContent = `${metaVar.name} (${metaVar.unit})`;
    document.getElementById('minVal').textContent = meta.min;
    document.getElementById('maxVal').textContent = meta.max;
    document.getElementById('legendGradient').style.background = GRADIENT_PRESETS[variable] || GRADIENT_PRESETS.temperature;
}

async function loadInstruments() {
    try {
        const res = await (await fetch(`${API}/api/instruments`)).json();
        const instruments = res.instruments;

        instrumentEntities.forEach(e => viewer.entities.remove(e));
        instrumentEntities = [];

        instruments.forEach(inst => {
            const pos = Cesium.Cartesian3.fromDegrees(inst.lon, inst.lat, 0);

            const entity = viewer.entities.add({
                name: 'instrument_' + inst.id,
                position: pos,
                point: {
                    pixelSize: 12,
                    color: Cesium.Color.fromCssColorString('#38bdf8'),
                    outlineColor: Cesium.Color.WHITE,
                    outlineWidth: 2,
                    scaleByDistance: new Cesium.NearFarScalar(1.5e2, 2.0, 1.5e7, 0.5),
                },
                label: {
                    text: inst.name,
                    font: '12px Inter, sans-serif',
                    fillColor: Cesium.Color.fromCssColorString('#e2e8f0'),
                    outlineColor: Cesium.Color.fromCssColorString('#0f172a'),
                    outlineWidth: 3,
                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                    pixelOffset: new Cesium.Cartesian2(0, -10),
                    show: true,
                },
            });

            const bottom = Cesium.Cartesian3.fromDegrees(inst.lon, inst.lat, -500);
            const line = viewer.entities.add({
                name: 'instrument_line_' + inst.id,
                polyline: {
                    positions: [pos, bottom],
                    width: 1,
                    material: Cesium.Color.fromCssColorString('#38bdf8').withAlpha(0.4),
                },
            });

            instrumentEntities.push(entity, line);
        });
    } catch (err) {
        console.error(err);
    }
}

async function loadVectors() {
    vectorEntities.forEach(e => viewer.entities.remove(e));
    vectorEntities = [];

    const show = document.getElementById('showCurrents').checked;
    if (!show) return;

    try {
        const depth = depths[document.getElementById('depth').value];
        const time = times[document.getElementById('time').value];

        const res = await (await fetch(`${API}/api/vectors?depth=${depth}&time=${time}&n=12&scale=0.5`)).json();

        res.vectors.forEach((v, i) => {
            const start = Cesium.Cartesian3.fromDegrees(v.lon, v.lat, -depth);
            const end = Cesium.Cartesian3.fromDegrees(v.end_lon, v.end_lat, -depth);
            const color = speedColor(v.speed);

            const arrow = viewer.entities.add({
                name: 'vector_' + i,
                polyline: {
                    positions: [start, end],
                    width: 2,
                    material: color,
                },
            });

            vectorEntities.push(arrow);
        });
    } catch (err) {
        console.error(err);
    }
}

function speedColor(speed) {
    const t = Math.min(1, speed / 1.2);
    const c = new Cesium.Color();
    c.setHSL(0.66 - t * 0.55, 1.0, 0.5);
    return c;
}

function handleClick(click) {
    const picked = viewer.scene.pick(click.position);
    const name = (Cesium.defined(picked) && Cesium.defined(picked.id)) ? (picked.id.name || '') : '';

    // Instrument click
    if (name.startsWith('instrument_')) {
        const id = name.replace('instrument_', '');
        selectedInstrument = id;
        loadProfile(id);
        hideProbe();
        return;
    }

    // Probe ocean data at clicked location
    probeAt(click.position);
}

function probeAt(screenPos) {
    if (!currentSlice) return;

    const cartesian = viewer.camera.pickEllipsoid(screenPos, viewer.scene.globe.ellipsoid);
    if (!cartesian) return;

    const cart = Cesium.Cartographic.fromCartesian(cartesian);
    const lon = Cesium.Math.toDegrees(cart.longitude);
    const lat = Cesium.Math.toDegrees(cart.latitude);

    const lons = currentSlice.lons;
    const lats = currentSlice.lats;

    if (lon < lons[0] || lon > lons[lons.length - 1] || lat < lats[0] || lat > lats[lats.length - 1]) {
        hideProbe();
        return;
    }

    let i = 0, j = 0;
    for (; i < lons.length - 1; i++) if (lons[i + 1] >= lon) break;
    for (; j < lats.length - 1; j++) if (lats[j + 1] >= lat) break;

    const row = currentSlice.values[j];
    const value = row ? row[i] : null;
    if (value == null) return;

    showProbe(lat, lon, value);
}

function showProbe(lat, lon, value) {
    const metaVar = variables.find(v => v.id === document.getElementById('variable').value);
    const depth = depths[document.getElementById('depth').value];
    const time = times[document.getElementById('time').value];

    const el = document.getElementById('probe');
    const content = document.getElementById('probeContent');
    content.innerHTML = `
        <div class="probe-row"><span class="probe-label">Variable</span><span class="probe-value">${metaVar.name}</span></div>
        <div class="probe-row"><span class="probe-label">Depth</span><span class="probe-value">${depth} m</span></div>
        <div class="probe-row"><span class="probe-label">Time</span><span class="probe-value">T${time}</span></div>
        <div class="probe-row"><span class="probe-label">Lat</span><span class="probe-value">${lat.toFixed(2)}°</span></div>
        <div class="probe-row"><span class="probe-label">Lon</span><span class="probe-value">${lon.toFixed(2)}°</span></div>
        <div class="probe-row"><span class="probe-label">Value</span><span class="probe-value">${value} ${metaVar.unit}</span></div>
    `;
    el.classList.remove('hidden');
}

function hideProbe() {
    document.getElementById('probe').classList.add('hidden');
}

async function loadProfile(instrumentId) {
    const variable = document.getElementById('variable').value;
    const time = times[document.getElementById('time').value];

    try {
        const res = await (await fetch(`${API}/api/profile/${instrumentId}?variable=${variable}&time=${time}`)).json();

        const metaVar = variables.find(v => v.id === variable);
        const depthsArr = res.profile.map(p => p.depth);
        const valuesArr = res.profile.map(p => p.value);

        document.getElementById('inst-name').textContent = `${res.instrument_id} — ${metaVar.name} at T${time}`;

        const ctx = document.getElementById('profileChart').getContext('2d');
        if (profileChart) profileChart.destroy();

        profileChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: `${metaVar.name} (${metaVar.unit})`,
                    data: depthsArr.map((d, i) => ({ x: valuesArr[i], y: d })),
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: metaVar.unit, color: '#94a3b8' },
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' },
                    },
                    y: {
                        reverse: true,
                        title: { display: true, text: 'Depth (m)', color: '#94a3b8' },
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' },
                    }
                },
                plugins: {
                    legend: { labels: { color: '#e2e8f0' } }
                }
            }
        });
    } catch (err) {
        console.error(err);
        showError('Failed to load instrument profile.');
    }
}

function setupEvents() {
    const varSel = document.getElementById('variable');
    const depthSlider = document.getElementById('depth');
    const timeSlider = document.getElementById('time');
    const playBtn = document.getElementById('playBtn');
    const showCurrents = document.getElementById('showCurrents');
    const showInstruments = document.getElementById('showInstruments');

    varSel.addEventListener('change', async () => {
        await loadDataLayer();
        if (selectedInstrument) loadProfile(selectedInstrument);
        await loadVectors();
    });

    depthSlider.addEventListener('input', () => {
        document.getElementById('depthVal').textContent = `${depths[depthSlider.value]} m`;
    });

    depthSlider.addEventListener('change', async () => {
        await loadDataLayer();
        await loadVectors();
    });

    timeSlider.addEventListener('input', () => {
        document.getElementById('timeVal').textContent = `T${times[timeSlider.value]}`;
    });

    timeSlider.addEventListener('change', async () => {
        await loadDataLayer();
        await loadVectors();
        if (selectedInstrument) loadProfile(selectedInstrument);
    });

    playBtn.addEventListener('click', togglePlay);

    showCurrents.addEventListener('change', async () => {
        await loadVectors();
    });

    showInstruments.addEventListener('change', () => {
        instrumentEntities.forEach(e => e.show = showInstruments.checked);
    });

    window.addEventListener('resize', () => {
        if (viewer) viewer.resize();
    });
}

function togglePlay() {
    const playBtn = document.getElementById('playBtn');
    const timeSlider = document.getElementById('time');

    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
        playBtn.textContent = 'Play';
        playBtn.classList.remove('active');
        return;
    }

    playBtn.textContent = 'Pause';
    playBtn.classList.add('active');

    playInterval = setInterval(() => {
        let idx = parseInt(timeSlider.value);
        idx = (idx + 1) % times.length;
        timeSlider.value = idx;
        document.getElementById('timeVal').textContent = `T${times[idx]}`;

        loadDataLayer().then(() => loadVectors()).then(() => {
            if (selectedInstrument) loadProfile(selectedInstrument);
        });
    }, 1200);
}

init();
