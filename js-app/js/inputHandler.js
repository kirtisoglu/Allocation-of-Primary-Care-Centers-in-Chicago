// Input handling
export class InputHandler {
    constructor(canvas, viewManager) {
        this.canvas = canvas;
        this.viewManager = viewManager;
        this.isDragging = false;
        this.dragStart = null;
    }

    findNearestNode(nodes, state, mousePos) {
        let nearest = null, minDist = 6 / state.transform.k;
        for (const n of nodes) {
            const d = Math.hypot(n.x - mousePos.x, n.y - mousePos.y);
            if (d < minDist) {
                minDist = d;
                nearest = n;
            }
        }
        return nearest;
    }

    renderTooltip(node, state, toleranceChecker, metadata) {
        const fmtInt = new Intl.NumberFormat('en-US');
        const fmtFixed = (x, k = 6) => (Number.isFinite(x) ? x.toFixed(k) : "N/A");

        const pop = node.population != null ? fmtInt.format(node.population) : "N/A";
        const within = toleranceChecker.isWithinTolerance(node, metadata) ? "✅ within" : "❌ outside";
        const isRoot = node.id === state.rootId ? " (root)" : "";
        const degree = state.links.filter(e => e.source === node.id || e.target === node.id).length;

        const overrideColor = state.nodeColorOverrides.get(node.id);
        const districtInfo = overrideColor ? `<div><b>District Color</b> <span style="display:inline-block;width:12px;height:12px;background:${overrideColor};border-radius:50%;vertical-align:middle;"></span></div>` : "";

        return `
            <div style="min-width: 250px; font-size: 12px; line-height: 1.6;">
                <div><b>Node ID</b> ${node.id}${isRoot}</div>
                <div><b>Lon</b> ${fmtFixed(node.x, 6)}</div>
                <div><b>Lat</b> ${fmtFixed(node.y, 6)}</div>
                <div><b>Degree</b> ${degree}</div>
                <div><b>Population:</b> ${pop} <span style="padding:1px 6px;border-radius:999px;background:#eee;font-size:11px;">${within}</span></div>
                <hr style="border: none; border-top: 1px solid #ddd; margin: 6px 0;">
                <div><b>Has Facility:</b> ${node.has_facility ? "✓" : "✗"}</div>
                <div><b>Compl. Facility:</b> ${node.compl_facility ? "✓" : "✗"}</div>
                <div><b>Candidate:</b> ${node.candidate ? "✓" : "✗"}</div>
                ${districtInfo}
            </div>
        `;
    }

    attachMouseListeners(canvas, state, viewManager, redraw, nodes, toleranceChecker, metadata, tooltip) {
        canvas.addEventListener("mousedown", e => {
            this.isDragging = true;
            this.dragStart = { x: e.clientX, y: e.clientY };
        });

        canvas.addEventListener("mousemove", e => {
            if (this.isDragging) {
                state.transform.x += e.clientX - this.dragStart.x;
                state.transform.y += e.clientY - this.dragStart.y;
                this.dragStart = { x: e.clientX, y: e.clientY };
                redraw();
            } else {
                const m = viewManager.getMousePos(e, state);
                const hit = this.findNearestNode(nodes, state, m);
                if (hit) {
                    tooltip.style.left = (e.pageX + 10) + "px";
                    tooltip.style.top = (e.pageY + 10) + "px";
                    tooltip.innerHTML = this.renderTooltip(hit, state, toleranceChecker, metadata);
                    tooltip.style.display = "block";
                } else {
                    tooltip.style.display = "none";
                }
            }
        });

        canvas.addEventListener("mouseup", () => {
            this.isDragging = false;
        });

        canvas.addEventListener("mouseleave", () => {
            this.isDragging = false;
            tooltip.style.display = "none";
        });

        canvas.addEventListener("wheel", e => {
            e.preventDefault();
            const scale = e.deltaY < 0 ? 1.1 : 0.9;
            const m = viewManager.getMousePos(e, state);
            state.transform.x -= m.x * (scale - 1) * state.transform.k;
            state.transform.y -= m.y * (scale - 1) * state.transform.k;
            state.transform.k *= scale;
            redraw();
        }, { passive: false });
    }
}