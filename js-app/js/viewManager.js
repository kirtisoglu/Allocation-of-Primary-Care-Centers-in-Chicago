// View management
export class ViewManager {
    constructor(canvas, config) {
        this.canvas = canvas;
        this.config = config;
    }

    autoCenterAndScale(state) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

        if (state.nodes.length) {
            for (const n of state.nodes) {
                if (Number.isFinite(n.x) && Number.isFinite(n.y)) {
                    minX = Math.min(minX, n.x);
                    minY = Math.min(minY, n.y);
                    maxX = Math.max(maxX, n.x);
                    maxY = Math.max(maxY, n.y);
                }
            }
        }

        if (state.blocksBounds) {
            minX = Math.min(minX, state.blocksBounds[0]);
            minY = Math.min(minY, state.blocksBounds[1]);
            maxX = Math.max(maxX, state.blocksBounds[2]);
            maxY = Math.max(maxY, state.blocksBounds[3]);
        }

        if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {
            return false;
        }

        if (!(minX < maxX && minY < maxY)) {
            return false;
        }

        const padding = this.config.framePadding;
        const width = Math.max(1e-9, maxX - minX);
        const height = Math.max(1e-9, maxY - minY);
        const scaleX = (this.canvas.width - 2 * padding) / width;
        const scaleY = (this.canvas.height - 2 * padding) / height;
        const scale = Math.min(scaleX, scaleY);

        state.center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
        state.transform.k = scale;
        state.transform.x = padding + (this.canvas.width - scale * (minX + maxX)) / 2;
        state.transform.y = padding + (this.canvas.height - scale * (minY + maxY)) / 2;
        state.transform.angle = 0;
        state.initialTransform = { ...state.transform };

        return true;
    }

    getMousePos(evt, state) {
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = evt.clientX - rect.left;
        const canvasY = evt.clientY - rect.top;

        const x = (canvasX - state.transform.x) / state.transform.k;
        const y = (canvasY - state.transform.y) / state.transform.k;
        const dx = x - state.center.x, dy = y - state.center.y;
        const cos = Math.cos(-state.transform.angle), sin = Math.sin(-state.transform.angle);
        let rx = dx * cos - dy * sin + state.center.x;
        let ry = dx * sin + dy * cos + state.center.y;
        if (state.flipX) rx = 2 * state.center.x - rx;
        return { x: rx, y: ry };
    }
}