// Data loading
import { GeometryUtils } from './geometry.js';

export class DataLoader {
    constructor(logger) {
        this.logger = logger;
    }

    async loadBlocks(blocksPaths, centroidMaps, blockIdToFeature, blockIdToGeometry) {
        try {
            this.logger.log("Loading blocks.geojson...");
            const response = await fetch("data/blocks.json");
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const gj = await response.json();

            if (!gj.features || !Array.isArray(gj.features)) {
                throw new Error("Invalid GeoJSON: missing features array");
            }

            const f0 = gj.features[0];
            let sample;
            if (f0?.geometry?.type === "Polygon") {
                sample = f0.geometry.coordinates?.[0]?.[0];
            } else if (f0?.geometry?.type === "MultiPolygon") {
                sample = f0.geometry.coordinates?.[0]?.[0]?.[0];
            }
            const detectedSwap = GeometryUtils.detectSwap(sample);
            this.logger.log(`Coordinate order: ${detectedSwap ? "LAT,LON" : "LON,LAT"}`);

            let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;

            const pushBounds = (ring) => {
                for (const pt of ring) {
                    const x = detectedSwap ? pt[1] : pt[0];
                    const y = detectedSwap ? pt[0] : pt[1];
                    minx = Math.min(minx, x);
                    miny = Math.min(miny, y);
                    maxx = Math.max(maxx, x);
                    maxy = Math.max(maxy, y);
                }
            };

            for (const f of gj.features) {
                const g = f.geometry;
                if (!g) continue;

                const fid = f.id != null ? String(f.id) : undefined;
                if (fid) {
                    blockIdToFeature.set(fid, f);
                    blockIdToGeometry.set(fid, g);
                }

                if (g.type === "Polygon") {
                    GeometryUtils.addPolygonPath(g.coordinates, detectedSwap, blocksPaths);
                    g.coordinates.forEach(pushBounds);
                } else if (g.type === "MultiPolygon") {
                    for (const poly of g.coordinates) {
                        GeometryUtils.addPolygonPath(poly, detectedSwap, blocksPaths);
                        poly.forEach(pushBounds);
                    }
                }

                const outer = g.type === "Polygon" ? g.coordinates[0]
                    : g.type === "MultiPolygon" ? g.coordinates[0][0] : null;
                if (outer && outer.length > 0) {
                    const centroid = GeometryUtils.calculateCentroid(outer, detectedSwap);
                    if (fid) centroidMaps.byFeatId.set(fid, centroid);
                    if (f.properties?.GEOID20) centroidMaps.byGeoID20.set(String(f.properties.GEOID20), centroid);
                    if (f.properties?.GEOID) centroidMaps.byGeoID.set(String(f.properties.GEOID), centroid);
                }
            }

            const bounds = (minx < maxx && miny < maxy) ? [minx, miny, maxx, maxy] : null;
            this.logger.log(`Blocks loaded: ${blocksPaths.length} polygons, ${centroidMaps.byFeatId.size} centroids`, "success");

            return { blocksBounds: bounds, detectedSwap };
        } catch (err) {
            this.logger.warn(`Failed to load blocks.geojson: ${err.message}`);
            throw err;
        }
    }

    async loadTree(iteration, centroidMaps) {
        try {
            this.logger.log(`Loading tree_${iteration}.json...`);
            const response = await fetch(`data/trees/tree_${iteration}.json`);
            if (!response.ok) {
                if (response.status === 404) {
                    this.logger.log(`Tree iteration ${iteration} not found`, "info");
                    return null;
                }
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            if (!data.nodes || !Array.isArray(data.nodes) || !data.links || !Array.isArray(data.links)) {
                throw new Error("Invalid tree JSON structure");
            }

            const metadata = data.metadata || null;
            const rootId = metadata?.root != null ? String(metadata.root) : null;

            const nodes = [];
            const missing = [];
            for (const n of data.nodes) {
                const idStr = String(n.id);
                const gid20 = n.GEOID20 ? String(n.GEOID20) : null;
                const gid = n.GEOID ? String(n.GEOID) : null;

                let c = null;
                if (gid20) c = centroidMaps.byGeoID20.get(gid20);
                if (!c && gid) c = centroidMaps.byGeoID.get(gid);
                if (!c) c = centroidMaps.byFeatId.get(idStr);
                if (!c && n.x != null && n.y != null) c = [n.x, n.y];

                if (!c) {
                    missing.push(idStr);
                    continue;
                }

                nodes.push({
                    id: idStr,
                    x: +c[0], // Use the determined centroid value
                    y: +c[1],
                    has_facility: n.has_facility ?? false,
                    compl_facility: n.compl_facility ?? false,
                    population: n.population ?? null,
                    candidate: n.candidate ?? false,
                });
            }

            if (missing.length > 0) {
                this.logger.warn(`${missing.length} nodes missing centroids`);
            }

            const nodesById = Object.fromEntries(nodes.map(n => [n.id, n]));
            const links = [];
            let unresolvedLinks = 0;
            for (const e of data.links) {
                const src = String(e.source), tgt = String(e.target);
                if (nodesById[src] && nodesById[tgt]) {
                    links.push({ source: src, target: tgt });
                } else {
                    unresolvedLinks++;
                }
            }

            this.logger.log(`Tree: ${nodes.length} nodes, ${links.length} links`, "success");
            return { nodes, links, nodesById, metadata, rootId };
        } catch (err) {
            this.logger.warn(`Failed to load tree_${iteration}.json: ${err.message}`);
            return null;
        }
    }

    async loadDistrict(iteration) {
        try {
            this.logger.log(`Loading district_${iteration}.json...`);
            const response = await fetch(`data/districts/district_${iteration}.json`);
            if (!response.ok) {
                if (response.status === 404) {
                    this.logger.log(`District iteration ${iteration} not found`, "info");
                    return null;
                }
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            if (!data.district || !Array.isArray(data.district)) {
                throw new Error("Invalid district JSON structure");
            }

            const nodes = data.district.map(id => String(id));
            const metadata = data.metadata || {};
            const color = this.generateDistrictColor(iteration);

            this.logger.log(`District: ${nodes.length} nodes`, "success");
            return { nodes, metadata, color };
        } catch (err) {
            this.logger.warn(`Failed to load district_${iteration}.json: ${err.message}`);
            return null;
        }
    }

    generateDistrictColor(iteration) {
        const hue = (iteration * 60) % 360;
        const saturation = 70 + (iteration % 3) * 10;
        const lightness = 50;
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }
}