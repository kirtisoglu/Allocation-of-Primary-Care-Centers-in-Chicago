// Animation control
export class AnimationController {
    constructor(dataLoader, logger) {
        this.dataLoader = dataLoader;
        this.logger = logger;
    }

    async play(state, redraw, viewManager, CONFIG) {
        state.isPlaying = true;
        state.isPaused = false;
        this.logger.log("Animation started", "success");
        await this.animationStep(state, redraw, viewManager, CONFIG);
    }

    pause(state) {
        state.isPlaying = false;
        state.isPaused = true;
        this.logger.log("Animation paused", "info");
    }

    stop(state, redraw) {
        state.isPlaying = false;
        state.isPaused = false;
        state.iteration = 0;
        state.nodes = [];
        state.links = [];
        state.nodesById = {};
        state.districts.clear();
        state.nodeColorOverrides.clear();
        state.metadata = null;
        state.rootId = null;
        this.logger.log("Animation stopped", "info");
        redraw();
    }

    async animationStep(state, redraw, viewManager, CONFIG) {
        if (!state.isPlaying) return;
    }
    async jumpToIteration(targetIter, state, redraw, viewManager, centroidMaps) {
        state.isPlaying = false;
        state.iteration = targetIter;
        this.logger.log(`Jumping to iteration ${targetIter}`, "info");

        const treeData = await this.dataLoader.loadTree(targetIter, centroidMaps);
        if (treeData) {
            state.nodes = treeData.nodes;
            state.links = treeData.links;
            state.nodesById = treeData.nodesById;
            state.metadata = treeData.metadata;
            state.rootId = treeData.rootId;

            viewManager.autoCenterAndScale(state);

            const districtData = await this.dataLoader.loadDistrict(targetIter);
            if (districtData) {
                for (const nodeId of districtData.nodes) {
                    state.nodeColorOverrides.set(nodeId, districtData.color);
                }
            }

            redraw();
        }
    }
}