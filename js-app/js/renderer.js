// Canvas rendering
export class Renderer {
    constructor(canvas, config, visual) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.config = config;
        this.visual = visual;
    }

    drawStarPath(x, y, rOuter = 3, spikes = 5, inset = 0.5) {
        const rot = Math.PI / 2 * 3;
        let step = Math.PI / spikes;
        this.ctx.beginPath();
        this.ctx.moveTo(x, y - rOuter);
        let rotA = rot;
        for (let i = 0; i < spikes; i++) {
            this.ctx.lineTo(x + Math.cos(rotA) * rOuter, y + Math.sin(rotA) * rOuter);
            rotA += step;
            this.ctx.lineTo(x + Math.cos(rotA) * (rOuter * inset), y + Math.sin(rotA) * (rOuter * inset));
            rotA += step;
        }
        this.ctx.closePath();
    }

    draw(state, isWithinTolerance) {
        const ctx = this.ctx;
        ctx.save();
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        ctx.translate(state.transform.x, state.transform.y);
        ctx.scale(state.transform.k, state.transform.k);
        ctx.translate(state.center.x, state.center.y);
        ctx.rotate(state.transform.angle);
        if (state.flipX) ctx.scale(-1, 1);
        ctx.translate(-state.center.x, -state.center.y);

        // Blocks with district colors (if districts are shown)
        if (state.showDistricts && state.districtBlockColors.size > 0) {
            for (const [blockId, color] of state.districtBlockColors.entries()) {
                const geom = state.blockIdToGeometry.get(blockId);
                if (!geom) continue;

                // Draw colored block
                if (geom.type === "Polygon") {
                    this.drawGeometry(ctx, geom.coordinates, color, state.transform.k);
                } else if (geom.type === "MultiPolygon") {
                    for (const poly of geom.coordinates) {
                        this.drawGeometry(ctx, poly, color, state.transform.k);
                    }
                }
            }
        }

        // Blocks (drawn first, behind everything)
        if (state.blocksPaths.length > 0) {
            ctx.fillStyle = this.config.colors.blockFill;
            ctx.strokeStyle = this.config.colors.blockStroke;
            ctx.lineWidth = this.visual.blockLineWidth / state.transform.k;
            for (const p of state.blocksPaths) {
                ctx.fill(p);
                ctx.stroke(p);
            }
        }

        // Links (if tree is shown)
        if (state.showTree && state.links.length > 0) {
            ctx.strokeStyle = this.config.colors.linkStroke;
            ctx.lineWidth = this.visual.linkLineWidth / state.transform.k;
            ctx.beginPath();
            for (const e of state.links) {
                const s = state.nodesById[e.source], t = state.nodesById[e.target];
                if (!s || !t) continue;
                ctx.moveTo(s.x, s.y);
                ctx.lineTo(t.x, t.y);
            }
            ctx.stroke();
        }

        // Nodes (if tree is shown)
        if (state.showTree && state.nodes.length > 0) {
            const now = Date.now();
            const isHighlighting = state.highlightNodeId && now < state.highlightUntil;
            const flashPhase = isHighlighting ? Math.sin((now / 100) * Math.PI * 2) : 0;

            for (const n of state.nodes) {
                const isRoot = n.id === state.rootId;
                const isHighlighted = isHighlighting && n.id === state.highlightNodeId;

                let color = null;
                if (state.nodeColorOverrides.has(n.id)) {
                    color = state.nodeColorOverrides.get(n.id);
                } else if (isRoot) {
                    color = this.config.colors.rootFill;
                } else {
                    color = isWithinTolerance(n) ? this.config.colors.greenFill : this.config.colors.redFill;
                }

                if (isRoot) {
                    const R = this.config.rootOuterPx / state.transform.k;
                    this.drawStarPath(n.x, n.y, R, 5, this.config.rootInset);
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.strokeStyle = this.config.colors.rootStroke;
                    ctx.lineWidth = this.visual.rootLineWidth / state.transform.k;
                    ctx.stroke();
                } else {
                    const r = this.config.nodeRadiusPx / state.transform.k;
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();

                    if (isHighlighted) {
                        ctx.strokeStyle = flashPhase > 0 ? `rgba(255, 255, 100, ${0.5 + flashPhase * 0.5})` : "transparent";
                        ctx.lineWidth = (2 + flashPhase * 2) / state.transform.k;
                        ctx.stroke();
                    } else if (!state.nodeColorOverrides.has(n.id) && isWithinTolerance(n)) {
                        ctx.strokeStyle = this.config.colors.greenStroke;
                        ctx.lineWidth = this.config.nodeStrokePx / state.transform.k;
                        ctx.stroke();
                    }
                }
            }
        }

        ctx.restore();
    }
    drawGeometry(ctx, coords, color, zoom) {
        const path = new Path2D();
        for (const ring of coords) {
            if (!ring?.length) continue;
            path.moveTo(ring[0][0], ring[0][1]);
            for (let i = 1; i < ring.length; i++) {
                path.lineTo(ring[i][0], ring[i][1]);
            }
            path.closePath();
        }
        ctx.fillStyle = color;
        ctx.fill(path);
        ctx.strokeStyle = "rgba(0, 0, 0, 0.3)";
        ctx.lineWidth = 1 / zoom;
        ctx.stroke(path);
    }
}