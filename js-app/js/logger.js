// Logging and status updates
export class Logger {
    constructor(statusPanel, config) {
        this.statusPanel = statusPanel;
        this.config = config;
    }

    log(msg, type = "info") {
        if (this.config.debug) {
            console.log(`[${type.toUpperCase()}] ${msg}`);
        }
        this.updateStatus(msg, type);
    }

    warn(msg) {
        console.warn(`[WARN] ${msg}`);
        this.updateStatus(msg, "error");
    }

    updateStatus(msg, type = "info") {
        const timestamp = new Date().toLocaleTimeString();
        let html = this.statusPanel.innerHTML;
        const maxLines = 8;
        const lines = html.split("<br>").slice(-(maxLines - 1));
        const color = type === "error" ? "#c62828" : type === "success" ? "#2e7d32" : "#666";
        this.statusPanel.innerHTML = [
            ...lines,
            `<span style="color: ${color}">[${timestamp}] ${msg}</span>`,
        ].join("<br>");
        this.statusPanel.scrollTop = this.statusPanel.scrollHeight;
    }
}