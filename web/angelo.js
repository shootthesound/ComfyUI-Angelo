// ComfyUI-Angelo — click-to-refine UI extension.
//
// Strategy: each AngeloRefine node gets its own canvas DOM widget
// attached at the bottom of the node. We draw the refined preview into
// that canvas ourselves (instead of using ComfyUI's auto-preview), and
// the canvas has a real DOM click listener. This sidesteps the issue
// where DOM image elements swallow clicks before LiteGraph sees them.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "AngeloRefine";

// Flip to true if something stops working and you want click → pixel-coord
// → widget-update → queue traced in the browser console.
const Angelo_DEBUG = false;
function dbg(...args) {
    if (Angelo_DEBUG) console.log("[Angelo]", ...args);
}


// --- Module-level hover tracking for the keyboard shortcuts. Set by
//     the canvas mouseenter / mouseleave handlers in attachPreviewCanvas.
let _AngeloHoveredNode = null;

// --- The node currently held in before/after compare via the '\' key.
//     Tracked at module level so the keyup / window-blur handlers can end
//     the compare even if the cursor has left the canvas mid-hold.
let _AngeloCompareNode = null;

// --- Global queuePrompt hook: bumps click_seq on every AngeloRefine
//     node where persistent_mask is on, so the standard ComfyUI Queue
//     button re-runs the refine on the held mask with a fresh seed.
//     Installed once per extension load; no-op for graphs without
//     AngeloRefine.
function installQueueHook() {
    if (app._AngeloQueueHookInstalled) return;
    if (typeof app.queuePrompt !== "function") return;
    const orig = app.queuePrompt.bind(app);
    app.queuePrompt = function (...args) {
        try {
            const nodes = (app.graph && app.graph._nodes) || [];
            // Angelo-initiated queues (a click/undo/re-roll/… on ONE node)
            // set _AngeloQueueInitiator to that node's id — the hook must
            // then leave every OTHER node alone, or editing node A re-runs
            // node B's held persistent mask. null = a real Queue press
            // (ComfyUI's button or the in-node ▶ Queue): bump all holders.
            const initiator = app._AngeloQueueInitiator;
            for (const n of nodes) {
                if (n?.type !== NODE_NAME) continue;
                if (initiator != null && n.id !== initiator) continue;
                const persistW = findWidget(n, "persistent_mask");
                if (!persistW || !persistW.value) continue;
                // A genuine Reset is in flight on this node — don't hijack
                // it. (Clearing `reset` here made Reset a silent no-op on
                // persistent-mask nodes; the 1s revert timer in triggerReset
                // already covers the stale-tick case this guarded against.)
                const resetW = findWidget(n, "reset");
                if (resetW && resetW.value) continue;
                const seqW = findWidget(n, "click_seq");
                if (!seqW) continue;
                setWidget(seqW, ((seqW.value || 0) + 1) & 0x7FFFFFFF);
                dbg("queueHook: bumped click_seq on persistent-mask node", n.id, "→", seqW.value);
            }
        } catch (e) {
            dbg("queueHook error (passing through)", e);
        }
        return orig(...args);
    };
    app._AngeloQueueHookInstalled = true;
    dbg("installed app.queuePrompt hook for persistent_mask");
}

// --- Global keyboard shortcuts. Active only when the cursor is hovering
//     a Angelo canvas AND that node is in Edit Mode. Mirrors
//     creative-tool conventions:
//       [ ]   → click_radius     (universal brush-size pattern)
//       { }   → feather_radius   (Photoshop brush hardness/softness)
//       , .   → denoise          (< > ordering on the same keys)
//     Captured in the capture phase so they beat ComfyUI's own
//     keybindings if any happen to overlap. preventDefault on a match
//     so the key event doesn't propagate further.
function installKeyboardShortcuts() {
    if (app._AngeloKeysInstalled) return;
    app._AngeloKeysInstalled = true;

    const bindings = [
        // [key, widget_name, delta, min, max, asInt, syncFn]
        ["[",  "click_radius",     -4,    8,   1024, true,  "syncClickRadiusInput"],
        ["]",  "click_radius",      4,    8,   1024, true,  "syncClickRadiusInput"],
        ["{",  "feather_radius",   -4,    0,    256, true,  "syncFeatherInput"],
        ["}",  "feather_radius",    4,    0,    256, true,  "syncFeatherInput"],
        [",",  "denoise",          -0.05, 0.05, 1.0, false, "syncDenoiseInput"],
        [".",  "denoise",           0.05, 0.05, 1.0, false, "syncDenoiseInput"],
    ];

    const handlers = {};
    for (const b of bindings) handlers[b[0]] = b;

    document.addEventListener("keydown", (event) => {
        let node = _AngeloHoveredNode;
        if (!node && event.key === "Escape") {
            // The Vary chooser / Outpaint review / detect overlays cover
            // the canvas, so the pointer's mouseleave already cleared the
            // hover — Esc must still reach them (the outpaint header even
            // advertises "Esc = cancel"). Fall back to whichever node owns
            // an open overlay.
            const nodes = (app.graph && app.graph._nodes) || [];
            node = nodes.find(n => n?.type === NODE_NAME
                && (isOutpaintReviewOpen(n) || isVaryChooserOpen(n)
                    || (n._AngeloDetections && n._AngeloDetections.length)));
        }
        if (!node) return;

        // Esc closes the Outpaint review first (cancels at zero cost —
        // nothing was committed).
        if (event.key === "Escape" && isOutpaintReviewOpen(node)) {
            hideOutpaintReview(node);
            event.preventDefault();
            return;
        }

        // Esc closes the Vary chooser first (keeps the current result),
        // before falling through to detect-mode dismissal.
        if (event.key === "Escape" && isVaryChooserOpen(node)) {
            hideVaryChooser(node);
            event.preventDefault();
            return;
        }

        // Esc dismisses pending detection candidates (even from an input,
        // so it works right after typing a concept + Detect).
        if (event.key === "Escape" && node._AngeloDetections && node._AngeloDetections.length) {
            clearDetections(node);
            return;
        }

        // Don't intercept when the user is typing in an input or textarea
        // (e.g., the toolbar Seed input, or any other DOM widget).
        const t = event.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
            return;
        }

        // Space also exits detect mode (only when not typing — handled
        // above — so a space in the concept box is unaffected).
        if (event.key === " " && node._AngeloDetections && node._AngeloDetections.length) {
            clearDetections(node);
            event.preventDefault();
            return;
        }

        // '\' (hold) = before/after compare — flash the session's BASE image
        // while held, release to return to the current state. Lightroom's
        // before/after key. Works in any mode (it's read-only); sits before
        // the Edit-Mode gate below. No modifiers so Shift+\ (|) stays free.
        if (event.key === "\\" && !event.ctrlKey && !event.metaKey && !event.altKey) {
            if (node._AngeloSourceImg && node._AngeloImg && !node._AngeloCompareHold) {
                node._AngeloCompareHold = true;
                _AngeloCompareNode = node;
                redrawCanvasWithOverlays(node);
            }
            event.preventDefault();
            event.stopPropagation();
            return;
        }

        // 'F' fits the image to the panel (zoom=1, centred). Works in any
        // mode — it mirrors the double-middle-click reset — so it sits before
        // the Edit-Mode gate below. No modifiers, so Ctrl-F etc. stay with the
        // browser.
        if ((event.key === "f" || event.key === "F")
            && !event.ctrlKey && !event.metaKey && !event.altKey) {
            if (node._AngeloImg) {
                resetView(node);
                redrawCanvasWithOverlays(node);
                event.preventDefault();
                event.stopPropagation();
            }
            return;
        }

        // Only active in Edit Mode. Sampler Mode has the toolbar
        // greyed; the keys would feel inert.
        const modeW = findWidget(node, "mode");
        if (!modeW || String(modeW.value) !== "Edit Mode") return;

        // NOTE: Undo/Redo are deliberately button-only (no Ctrl-Z / Ctrl-Y /
        // Ctrl-Shift-Z). Binding those over the canvas clashed too much with
        // ComfyUI's graph-level undo/redo, so the shortcuts were removed.

        const binding = handlers[event.key];
        if (!binding) return;

        const [, name, delta, min, max, asInt, syncName] = binding;
        const w = findWidget(node, name);
        if (!w) return;
        let v = Number(w.value || 0) + delta;
        v = Math.max(min, Math.min(max, v));
        if (asInt) v = Math.round(v);
        else v = Math.round(v * 100) / 100;  // avoid float drift accumulating
        setWidget(w, v);

        // Sync the corresponding toolbar input so the visible value
        // tracks the keyboard adjustment.
        const syncFn = ({
            syncClickRadiusInput, syncFeatherInput,
            syncDenoiseInput,
        })[syncName];
        if (syncFn) syncFn(node);

        // For click_radius, also redraw the canvas so the hover ring
        // resizes immediately on the visible image.
        if (name === "click_radius" && typeof redrawCanvasWithOverlays === "function") {
            redrawCanvasWithOverlays(node);
        }

        dbg("key", event.key, "→", name, "=", v);
        event.preventDefault();
        event.stopPropagation();
    }, true);  // capture phase

    // End the '\' before/after compare on key release — or on window blur,
    // which swallows the keyup (alt-tab mid-hold would otherwise leave the
    // BEFORE view stuck).
    const endCompare = () => {
        const n = _AngeloCompareNode;
        if (!n) return;
        n._AngeloCompareHold = false;
        _AngeloCompareNode = null;
        redrawCanvasWithOverlays(n);
    };
    document.addEventListener("keyup", (event) => {
        if (event.key === "\\") endCompare();
    }, true);
    window.addEventListener("blur", endCompare);

    // Handle image paste (Ctrl+V / Cmd+V) from the OS clipboard.
    window.addEventListener("paste", (event) => {
        const node = _AngeloHoveredNode;
        if (!node) return; // Only active when hovering over an Angelo canvas.

        // Do not intercept if the user is pasting text into an input or textarea.
        const activeTag = document.activeElement ? document.activeElement.tagName : "";
        const isEditable = document.activeElement && document.activeElement.isContentEditable;
        if (activeTag === "INPUT" || activeTag === "TEXTAREA" || isEditable) {
            return;
        }

        if (event.clipboardData && event.clipboardData.files && event.clipboardData.files.length > 0) {
            const imageFiles = Array.from(event.clipboardData.files).filter(f => f.type.startsWith("image/"));
            
            if (imageFiles.length > 0) {
                // Strictly prevent ComfyUI from intercepting this and spawning a LoadImage node
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                
                let file = imageFiles[0];
                
                // Browsers often name pasted files generically like "image.png".
                // Append a timestamp to make it distinct in uploads and logs.
                if (file.name === "image.png" || !file.name) {
                    const ext = file.type.split('/')[1] || "png";
                    const fakeName = `pasted_${Date.now()}.${ext}`;
                    file = new File([file], fakeName, { type: file.type });
                }
                
                // Route it through the same resolution popup as Drag & Drop / Load Image.
                showLoadImagePopup(node, file);
            }
        }
    }, true);  // capture phase on window guarantees it fires before ComfyUI's document listener
    
    dbg("installed keyboard shortcuts");
}

// --- One-shot migration for workflows saved on v2.0.0–v2.1.0. ---
// Those versions had a `lir_seq` widget between `upscale_seq` and
// `quick_prompt_mode`; v2.2.0 removed it, shifting every later positional
// widgets_values slot left by one on load. Old widget tail:
//     [..., upscale_seq, lir_seq, quick_prompt_mode]        (save)
//     [..., upscale_seq, quick_prompt_mode, shrink_seq, …]  (load target)
// so exactly TWO widgets land wrong: quick_prompt_mode receives the dead
// lir_seq INT (→ combo validation fails at Queue time), and shrink_seq
// receives the old quick_prompt_mode STRING. Everything after got no value
// (the old array was shorter) and already holds its default.
//
// Detection is SEMANTIC, not version-based: quick_prompt_mode is a combo
// whose serialized value is always a string in a healthy save — a number
// there is an unambiguous marker of the pre-2.2 layout. Healthy saves are
// a no-op, and the next save serializes the repaired values, so the shim
// self-deactivates. (Name-keyed widgets_values formats never shift and
// never trip the marker.)
function migrateLegacyWidgetValues(node) {
    const qpW = findWidget(node, "quick_prompt_mode");
    if (!qpW || typeof qpW.value !== "number") return;   // healthy save
    const ssW = findWidget(node, "shrink_seq");
    const validModes = (qpW.options && qpW.options.values) || [];
    if (ssW && typeof ssW.value === "string" && validModes.includes(ssW.value)) {
        // The real prompt-mode string slid one slot down onto shrink_seq —
        // recover it.
        qpW.value = ssW.value;
    } else {
        // v2.0.0-era save (no quick_prompt_mode yet) — default it.
        qpW.value = "Identity + Quality";
    }
    if (ssW && typeof ssW.value !== "number") ssW.value = 0;
    console.log("[Angelo] migrated a pre-2.2 workflow save (removed lir_seq "
        + "slot) — ✨ prompt mode restored to \"" + qpW.value + "\"");
}

// --- Generic type-sanity sweep for ANY misaligned old save (#33). ---
// Positional widgets_values from sufficiently old versions (v1.8 etc.) can
// land values on the wrong widgets in ways the specific shim above doesn't
// recognise — e.g. an old text widget's "" arriving on the outpaint_seq INT,
// which then fails ComfyUI's server-side prompt validation ("invalid literal
// for int()") before any of our Python runs. This sweep doesn't try to
// recover meaning; it guarantees TYPE validity so the node can queue at all:
// number widgets must hold finite numbers, combos must hold a listed option,
// booleans must hold booleans. Anything else resets to the widget's
// definition default (snapshotted in onNodeCreated, which runs before the
// serialized values are restored). Worst case equals the manual
// right-click → Fix node (recreate) workaround, minus the manual step.
// Healthy saves are a strict no-op. MUST run AFTER migrateLegacyWidgetValues
// — the specific shim recovers real values and its detection marker (a
// number on the quick_prompt_mode combo) is exactly the kind of state this
// sweep would otherwise reset to a default first.
function sanitizeWidgetTypes(node) {
    const defaults = node._AngeloWidgetDefaults;
    if (!defaults || !node.widgets) return;
    const repaired = [];
    for (const w of node.widgets) {
        if (!w || !(w.name in defaults)) continue;   // DOM/custom widgets: skip
        const d = defaults[w.name];
        const comboValues = (w.options && Array.isArray(w.options.values))
            ? w.options.values : null;
        if (comboValues) {
            if (!comboValues.includes(w.value)) {
                w.value = comboValues.includes(d) ? d : comboValues[0];
                repaired.push(w.name);
            }
        } else if (typeof d === "number") {
            if (typeof w.value !== "number" || !Number.isFinite(w.value)) {
                // A numeric STRING is a plausibly-real value (some formats
                // stringify numbers) — keep it. "" / null / junk → default.
                const n = (typeof w.value === "string" && w.value.trim() !== "")
                    ? Number(w.value) : NaN;
                w.value = Number.isFinite(n) ? n : d;
                repaired.push(w.name);
            }
        } else if (typeof d === "boolean") {
            if (typeof w.value !== "boolean") {
                w.value = d;
                repaired.push(w.name);
            }
        } else if (typeof d === "string") {
            if (typeof w.value !== "string") {
                w.value = String(w.value == null ? "" : w.value);
                repaired.push(w.name);
            }
        }
    }
    if (repaired.length) {
        console.log("[Angelo] repaired " + repaired.length + " misaligned widget "
            + "value(s) from an older save (" + repaired.join(", ") + ") — the "
            + "node can queue again. If any toolbar setting looks off, "
            + "right-click → Fix node (recreate) for a fully fresh layout.");
    }
}

app.registerExtension({
    name: "Angelo.ClickToRefine",

    async setup() {
        installQueueHook();
        installKeyboardShortcuts();
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_NAME) return;

        // --- Node setup: attach the preview canvas + hide mechanical widgets ---
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            // Snapshot every widget's definition default BEFORE a workflow
            // load restores serialized values (onConfigure runs after this)
            // and BEFORE attachPreviewCanvas adds the DOM widget — feeds
            // the sanitizeWidgetTypes sweep for misaligned old saves (#33).
            this._AngeloWidgetDefaults = {};
            if (this.widgets) {
                for (const w of this.widgets) {
                    if (w && w.name != null) this._AngeloWidgetDefaults[w.name] = w.value;
                }
            }
            hideMechanicalWidgets(this);
            attachPreviewCanvas(this);
            // Force LiteGraph to recompute layout now that hidden widgets
            // claim zero height — otherwise the node keeps its initial
            // tall size and the (now-hidden) widget slots show as gaps.
            if (typeof this.setSize === "function" && this.computeSize) {
                const min = this.computeSize();
                if (this.size[1] < min[1]) this.size[1] = min[1];
            }
            // Reflect persisted widget state in every toolbar control.
            // NOTE: on an existing-workflow load this runs BEFORE the
            // serialized widget values are restored, so the toggles
            // may show defaults here. The onConfigure hook below re-runs
            // the same sync after the restore to correct any mismatch.
            syncAllToolbarControls(this);

            // Mode widget: grey toolbar in Sampler Mode, un-grey in
            // Refinement; auto-force sampler_seed_control = fixed when
            // flipping into Refinement (and restore sampler_seed to its
            // at-run value via lockSeedToAtRun).
            const modeW = findWidget(this, "mode");
            if (modeW) {
                const origCb = modeW.callback;
                modeW.callback = (value, ...args) => {
                    const prevValue = modeW._AngeloPrevValue;
                    if (origCb) {
                        try { origCb.call(modeW, value, ...args); }
                        catch (e) { dbg("mode callback orig threw", e); }
                    }
                    modeW._AngeloPrevValue = value;
                    syncModeSwitchToFixed(this, prevValue);
                    syncModeState(this);
                };
                modeW._AngeloPrevValue = modeW.value;
            }
            syncModeState(this);   // initial state reflects persisted widget

            // Seed-control widgets: when value transitions TO "fixed"
            // (either by user click or programmatic set), restore the
            // corresponding seed widget to the seed_at_run value. This
            // ensures "fixed" always means "the seed that produced the
            // current canvas", not whatever value after-gen left in the
            // widget. Wrap each widget's callback to detect the transition.
            const samplerCtrlW = findWidget(this, "sampler_seed_control");
            if (samplerCtrlW) {
                const origCb = samplerCtrlW.callback;
                samplerCtrlW.callback = (value, ...args) => {
                    const prevValue = samplerCtrlW._AngeloPrevValue;
                    if (origCb) {
                        try { origCb.call(samplerCtrlW, value, ...args); }
                        catch (e) { dbg("sampler_seed_control callback orig threw", e); }
                    }
                    samplerCtrlW._AngeloPrevValue = value;
                    if (value === "fixed" && prevValue !== "fixed") {
                        lockSeedToAtRun(this, "sampler_seed", "sampler_seed_control");
                    }
                };
                samplerCtrlW._AngeloPrevValue = samplerCtrlW.value;
            }

            const seedCtrlW = findWidget(this, "seed_control");
            if (seedCtrlW) {
                const origCb = seedCtrlW.callback;
                seedCtrlW.callback = (value, ...args) => {
                    const prevValue = seedCtrlW._AngeloPrevValue;
                    if (origCb) {
                        try { origCb.call(seedCtrlW, value, ...args); }
                        catch (e) { dbg("seed_control callback orig threw", e); }
                    }
                    seedCtrlW._AngeloPrevValue = value;
                    if (value === "fixed" && prevValue !== "fixed") {
                        lockSeedToAtRun(this, "seed", "seed_control");
                    }
                };
                seedCtrlW._AngeloPrevValue = seedCtrlW.value;
            }
            const min = [340, 540];
            if (this.size[0] < min[0]) this.size[0] = min[0];
            if (this.size[1] < min[1]) this.size[1] = min[1];
        };

        // --- Right-click node menu: Open / Copy / Paste image (#7). Covers
        //     right-clicks on the litegraph-rendered node body / title; the
        //     DOM preview canvas has its own contextmenu handler too. ---
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            if (origGetExtraMenuOptions) origGetExtraMenuOptions.apply(this, arguments);
            const node = this;
            const hasImg = !!node._AngeloImg;
            const inSampler = String(findWidget(node, "mode")?.value) === "Sampler Mode";
            options.unshift(
                // Quick actions (#29) — same set as the preview canvas menu.
                { content: inSampler ? "Angelo: switch to Edit Mode"
                                     : "Angelo: switch to Sampler Mode",
                  callback: () => _angeloSetMode(node, inSampler ? "Edit Mode" : "Sampler Mode") },
                { content: "Angelo: generate new base (fresh seed)",
                  callback: () => _angeloGenerateNewBase(node) },
                { content: "Angelo: regenerate same base",
                  callback: () => _angeloRegenerateSameBase(node) },
                { content: "Angelo: open image in new tab", disabled: !hasImg,
                  callback: () => _angeloOpenImageInTab(node) },
                { content: "Angelo: copy image", disabled: !hasImg,
                  callback: () => _angeloCopyImageToClipboard(node) },
                null,
            );
            return options;
        };

        // --- onExecuted: receive the new preview URL, draw into our canvas,
        //     and run after-gen seed control + record seed_at_run for the
        //     lock-on-fixed semantics. ---
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            dbg("onExecuted", message);

            // Preview image (Angelo_preview)
            const refs = message?.Angelo_preview;
            if (refs && refs.length > 0) {
                const ref = refs[0];
                this._AngeloPreviewRef = ref;   // for the SAM 3 / YOLO detect route
                const url = makeViewUrl(ref);
                dbg("loading preview", url);
                loadIntoCanvas(this, url);
            }

            // Base image for the hold-to-compare key (\). Preloaded into an
            // off-canvas Image so the flash is instant; only re-fetched when
            // the base actually changes (Python reuses the cached refs).
            const srcRefs = message?.Angelo_source_preview;
            if (srcRefs && srcRefs.length > 0) {
                const srcUrl = makeViewUrl(srcRefs[0]);
                if (this._AngeloSourceUrl !== srcUrl) {
                    this._AngeloSourceUrl = srcUrl;
                    const node = this;
                    const im = new Image();
                    im.crossOrigin = "anonymous";
                    im.onload = () => { node._AngeloSourceImg = im; };
                    im.onerror = () => dbg("source preview load error", srcUrl);
                    im.src = srcUrl;
                }
            }

            // NOTE: seg_polygon is intentionally NOT cleared here. It must
            // persist across runs like stroke_points / rect_points so a
            // Persistent Mask re-roll keeps using the SAM-detected
            // silhouette. It's cleared instead by the manual-mask triggers
            // (a click / paint / rect resets it) — see triggerRefine etc.

            // Auto-switched base generation: Python ran Sampler Mode because
            // the workflow was queued in Edit Mode with no session and a
            // blank wired latent (fresh open / ComfyUI restart). Flip the
            // Mode widget so the UI matches what actually ran — before the
            // after-gen block below, which then applies to the sampler seed
            // group as it should for a base generation.
            if (message?.Angelo_auto_sampler?.[0]) {
                _angeloSetMode(this, "Sampler Mode");
            }

            // Seed_at_run capture — used by the lock-on-fixed code. ComfyUI's
            // ui message values arrive as 1-element lists (their convention).
            const lastMode = message?.Angelo_mode?.[0];
            const samplerSeedAtRun = message?.Angelo_sampler_seed_at_run?.[0];
            const refineSeedAtRun = message?.Angelo_refine_seed_at_run?.[0];
            if (samplerSeedAtRun != null) {
                this._AngeloSamplerSeedAtRun = Number(samplerSeedAtRun);
            }
            if (refineSeedAtRun != null) {
                this._AngeloRefineSeedAtRun = Number(refineSeedAtRun);
            }

            // After-gen seed control. ComfyUI's standard "seed widgets" have
            // an auto-added control_after_generate dropdown that does this
            // for them; ours are explicit ENUM widgets, so we apply the
            // logic ourselves. Runs AFTER seed_at_run is captured so a
            // subsequent lock-on-fixed restores the pre-modification value.
            if (lastMode === "Sampler Mode") {
                applyAfterGenControl(this, "sampler_seed", "sampler_seed_control");
            } else if (lastMode === "Edit Mode") {
                applyAfterGenControl(this, "seed", "seed_control");
            }

            // Vary ×4: candidate previews arrived → open the chooser.
            const varyRefs = message?.Angelo_vary_previews;
            if (varyRefs && varyRefs.length) {
                showVaryChooser(this, varyRefs);
            }

            // Outpaint: the extended canvas arrived → open the review.
            const opRefs = message?.Angelo_outpaint_preview;
            if (opRefs && opRefs.length) {
                showOutpaintReview(this, opRefs[0]);
            }

            // Fix All: this run has landed — fire the next candidate. The
            // small delay lets the fresh preview paint (and the green tick
            // register) before the next confirm queues.
            if (this._AngeloFixAll) {
                const node = this;
                setTimeout(() => _angeloFixAllStep(node), 120);
            }
        };

        // Reset and Undo now live on the DOM toggle bar above the canvas
        // (see attachPreviewCanvas). Removed the canvas-title-bar hooks
        // that used to draw + hit-test them — all interactive controls
        // sit on one horizontal line.

        // --- onConfigure: fires when a saved workflow is loaded and the
        //     node's serialized widget values are restored. onNodeCreated
        //     runs BEFORE that restore, so the toolbar would otherwise
        //     reflect defaults (e.g. Paint Mode shows OFF when the saved
        //     value was ON, and vice-versa). Re-run the full toolbar sync
        //     here so the DOM controls match the restored widget state. ---
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            // Defer one tick — some ComfyUI versions finish applying
            // widget values immediately after onConfigure returns, so a
            // microtask-delayed sync sees the final restored state.
            const node = this;
            queueMicrotask(() => {
                try { migrateLegacyWidgetValues(node); }
                catch (e) { dbg("legacy widget migration threw", e); }
                // AFTER the specific shim (it recovers real values; this
                // one guarantees type validity for anything else) — #33.
                try { sanitizeWidgetTypes(node); }
                catch (e) { dbg("widget type sweep threw", e); }
                try { syncAllToolbarControls(node); }
                catch (e) { dbg("onConfigure sync threw", e); }
            });
        };
    },
});


// ============================================================
// Preview canvas (the DOM widget that displays + handles clicks)
// ============================================================

function attachPreviewCanvas(node) {
    if (node._AngeloWidget) return;  // already attached

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.position = "relative";
    container.style.background = "#1a1a1a";
    container.style.border = "1px solid #333";
    container.style.borderRadius = "4px";
    container.style.overflow = "hidden";
    container.style.display = "flex";
    container.style.flexDirection = "column";

    // Toolbar above the canvas, top to bottom:
    //   modeRow     — Mode dropdown, centred (the master Sampler/Edit switch)
    //   row3        — shared generation config (steps/cfg/sampler/sched), always active
    //   row4        — Sampler-Mode base-gen seed group, greyed in Edit Mode
    //   refineRows  — edit actions + refine values, greyed in Sampler Mode
    const toggleBarWrap = document.createElement("div");
    toggleBarWrap.style.display = "flex";
    toggleBarWrap.style.flexDirection = "column";
    toggleBarWrap.style.background = "#222";
    toggleBarWrap.style.borderBottom = "1px solid #333";
    toggleBarWrap.style.transition = "opacity 0.15s ease";  // smooth grey/un-grey on mode switch
    node._AngeloToolbarWrap = toggleBarWrap;

    // Mode row — centred at the top of the node, visually separated from
    // the rest of the toolbar by a bottom border.
    const modeRow = makeToolbarRow();
    modeRow.style.justifyContent = "center";
    modeRow.style.borderBottom = "1px solid #333";
    modeRow.style.position = "relative";  // anchor for the floating Cancel Detect button

    // Generation / sampler-seed rows (set-once-ish, sit under Mode).
    const row3 = makeToolbarRow();
    const row4 = makeToolbarRow();
    row4.style.transition = "opacity 0.15s ease";

    // Edit control rows — greyed in Sampler Mode (wrapped so the grey
    // target is just these rows, not the generation rows). A top border
    // separates the "edit" group from the "generation" group above.
    const refineRowsWrap = document.createElement("div");
    refineRowsWrap.style.transition = "opacity 0.15s ease";
    refineRowsWrap.style.borderTop = "1px solid #333";
    const row1 = makeToolbarRow();
    const row2 = makeToolbarRow();
    const outpaintRow = makeToolbarRow(); // Outpaint mode only (populated below)
    outpaintRow.style.display = "none";
    const quickRow = makeToolbarRow();    // Quick Actions (one-press magic buttons)
    const detectRow = makeToolbarRow();   // SAM 3 detect (Refine + Smart Inpaint)
    detectRow.style.flexWrap = "nowrap";  // keep it one line; the text box flexes
    refineRowsWrap.appendChild(row1);
    refineRowsWrap.appendChild(row2);
    refineRowsWrap.appendChild(outpaintRow);
    refineRowsWrap.appendChild(quickRow);
    refineRowsWrap.appendChild(detectRow);
    node._AngeloOutpaintRow = outpaintRow;
    node._AngeloQuickRow = quickRow;
    node._AngeloDetectRow = detectRow;

    toggleBarWrap.appendChild(modeRow);
    toggleBarWrap.appendChild(row3);
    toggleBarWrap.appendChild(row4);
    toggleBarWrap.appendChild(refineRowsWrap);

    node._AngeloModeRow = modeRow;
    node._AngeloRefineRowsWrap = refineRowsWrap;
    node._AngeloSamplerSeedRow = row4;

    // ===== ROW 1: actions + mode toggles =====
    const resetBtn = makeActionButton("Reset", () => triggerReset(node), "reset");
    resetBtn.title = "Throw away the cached refined latent + history and start fresh from the upstream latent.";
    row1.appendChild(resetBtn);

    // Undo / Redo as a joined two-icon pair — they're conceptually one
    // control, and the glyphs (with full-word tooltips) reclaim row
    // width for Re-roll / Vary ×4.
    const undoRedoWrap = document.createElement("div");
    undoRedoWrap.style.cssText = "display:inline-flex; flex:0 0 auto;";
    const undoBtn = makeActionButton("⟲", () => triggerUndo(node), "undo");
    undoBtn.title = "Undo — pop the most recent refine off the history stack (up to 10 deep). Restores the cached latent from before the last edit.";
    undoBtn.style.borderRadius = "3px 0 0 3px";
    undoBtn.style.padding = "3px 9px";
    undoBtn.style.fontSize = "13px";
    undoBtn.style.lineHeight = "15px";
    const redoBtn = makeActionButton("⟳", () => triggerRedo(node), "redo");
    redoBtn.title = "Redo — re-apply the most recent edit that Undo removed. A new edit clears the redo history.";
    redoBtn.style.borderRadius = "0 3px 3px 0";
    redoBtn.style.borderLeft = "none";
    redoBtn.style.padding = "3px 9px";
    redoBtn.style.fontSize = "13px";
    redoBtn.style.lineHeight = "15px";
    undoRedoWrap.appendChild(undoBtn);
    undoRedoWrap.appendChild(redoBtn);
    row1.appendChild(undoRedoWrap);

    const rerollBtn = makeActionButton("Re-roll", () => triggerReroll(node), "reroll");
    rerollBtn.title = "Try the most recent edit again with a fresh seed — SAME mask, SAME starting "
        + "image. Each press replaces the last attempt with a new variation (it doesn't stack on "
        + "top). Make an edit first, then Re-roll to cycle seeds without re-painting or resetting. "
        + "Works for clicks, brush strokes, rectangles and detected masks.\n\n"
        + "After a ✨ Quick Photo Refine, Re-roll re-runs the ✨ pass instead (same as pressing ✨ "
        + "again) — it always re-rolls the last thing you did. After an Undo/Redo it's inert until "
        + "your next edit.";
    row1.appendChild(rerollBtn);
    node._AngeloRerollBtn = rerollBtn;

    const varyBtn = makeActionButton("Vary ×4", () => triggerVary(node), "vary");
    varyBtn.title = "Generate FOUR variations of the most recent edit at once — same mask, same "
        + "starting image, four different seeds — then pick your favourite from a chooser overlay. "
        + "Re-roll with four dice instead of one. The chosen variation replaces the last attempt "
        + "(✕ on the chooser keeps the current one); nothing commits until you pick. Make an edit "
        + "first. Not available while Restore is ON (restores are deterministic).";
    row1.appendChild(varyBtn);
    node._AngeloVaryBtn = varyBtn;


    row1.appendChild(makeSeparator());

    const persistentMaskToggle = makeToggleButton("Persistent Mask", () => {
        const w = findWidget(node, "persistent_mask");
        if (!w) return;
        setWidget(w, !w.value);
        syncPersistentMaskToggle(node);
    });
    persistentMaskToggle.title = "When ON, the last mask is held. Pressing the standard ComfyUI Queue button re-runs that region on the LATEST result with a fresh seed each time, so each press builds further — gradually morph an area over several presses without re-painting. (To re-roll the same edit on the ORIGINAL image instead, use the Re-roll button.)";
    row1.appendChild(persistentMaskToggle);
    node._AngeloPersistentMaskToggle = persistentMaskToggle;

    const areaPromptToggle = makeToggleButton("Area Prompt", () => {
        const w = findWidget(node, "area_prompt");
        if (!w) return;
        setWidget(w, !w.value);
        syncAreaPromptToggle(node);
        syncAreaPromptVisibility(node);
    });
    areaPromptToggle.title = "When ON, a text box appears between the toolbar and the canvas. Refines encode that text with the connected CLIP and use it instead of the main prompt — paint/drag a region and reshape it with a different prompt. Requires a CLIP input wired + non-empty area text. Forced ON in Smart Inpaint.";
    row1.appendChild(areaPromptToggle);
    node._AngeloAreaPromptToggle = areaPromptToggle;

    const paintModeToggle = makeToggleButton("Paint Mode", () => {
        const w = findWidget(node, "paint_mode");
        if (!w) return;
        setWidget(w, !w.value);
        syncPaintModeToggle(node);
    });
    paintModeToggle.title = "When ON, hold + drag on the preview paints a freeform brush stroke (each dragged point becomes a circle of click_radius; the union is the refine mask). Release to submit. When OFF, clicks behave as single-circle refines.";
    row1.appendChild(paintModeToggle);
    node._AngeloPaintModeToggle = paintModeToggle;

    // Restore / Remove are conflicting brushes — turning one on turns the other
    // off so the toolbar can't lie about which is active.
    //
    // Remove REQUIRES Xtra-Fine, so toggling remove_mode also forces
    // fine_upscaling on (remembering the prior state) and restores it on the
    // way out — off, unless Xtra-Fine was already on before Remove was
    // enabled. The remembered state lives in node.properties (NOT a plain JS
    // field): LiteGraph serializes properties with the workflow, so a save
    // made with Remove ON still restores Xtra-Fine correctly after a reload.
    const _setRemove = (on) => {
        const w = findWidget(node, "remove_mode");
        if (!w) return;
        const fw = findWidget(node, "fine_upscaling");
        node.properties = node.properties || {};
        if (on && !w.value) {
            node.properties.angelo_xf_before_remove = fw ? !!fw.value : false;
            if (fw) setWidget(fw, true);
        } else if (!on && w.value) {
            if (fw) setWidget(fw, !!node.properties.angelo_xf_before_remove);
            delete node.properties.angelo_xf_before_remove;
        }
        setWidget(w, on);
    };
    const _clearOtherBrushes = (keep) => {
        if (keep !== "restore_mode") {
            const rw = findWidget(node, "restore_mode");
            if (rw && rw.value) setWidget(rw, false);
        }
        if (keep !== "remove_mode") _setRemove(false);   // also restores Xtra-Fine
    };
    const _syncBrushToggles = () => {
        syncRestoreToggle(node);
        syncRemoveToggle(node);
        syncFineUpscaleToggle(node);
    };

    const restoreToggle = makeToggleButton("Restore", () => {
        const w = findWidget(node, "restore_mode");
        if (!w) return;
        const next = !w.value;
        setWidget(w, next);
        if (next) _clearOtherBrushes("restore_mode");
        _syncBrushToggles();
    });
    restoreToggle.title = "Restore brush — when ON, clicks and paint strokes RESTORE the painted "
        + "region back to the session's original base image instead of refining it. No sampling at "
        + "all, so it's instant. Use it to bring back details an edit shouldn't have touched (e.g. "
        + "refine a spacesuit, then brush the face inside the helmet back to the original). Feather "
        + "applies for a soft blend; works with clicks, Paint Mode strokes, and Detect masks. "
        + "Refine mode only — the Smart modes ignore it.";
    row1.appendChild(restoreToggle);
    node._AngeloRestoreToggle = restoreToggle;

    // Remove brush — sits right beside Restore because it's the same gesture
    // with a different fill: Restore brings the ORIGINAL back, Remove erases
    // to BACKGROUND. Paint / click / Detect a region, and the edit model fills
    // the hole with a continuation of the surroundings. Edit models only.
    const removeToggle = makeToggleButton("🩹 Remove", () => {
        const w = findWidget(node, "remove_mode");
        if (!w) return;
        const next = !w.value;
        if (next) _clearOtherBrushes("remove_mode");
        _setRemove(next);   // toggles remove_mode + forces/restores Xtra-Fine
        _syncBrushToggles();
    });
    removeToggle.title = "Remove brush — erase an object and let the edit model rebuild the "
        + "background behind it. Paint over it (or click, or Detect \"the person\" and it uses that "
        + "mask). One full-denoise pass regenerates the masked region as background: the object is "
        + "cut out of the reference the model anchors to (a real hole), so it rebuilds from the "
        + "surroundings instead of redrawing the object.\n\n"
        + "The Denoise box is ignored (the pass is always full denoise); your main/Area prompt isn't "
        + "used either — a fixed background-fill instruction drives it. Feather is forced to 0 and the mask "
        + "auto-grows ~10% to cover the object's halo.\n\n"
        + "Remove auto-enables Xtra-Fine (and shows Ctx Pad / MP / Method) and requires it — the "
        + "removal always runs on a high-res crop, which is what rebuilds fine background detail. "
        + "Turning Remove off restores Xtra-Fine to how it was before.\n\n"
        + "Edit models only (FLUX 2 Klein / Qwen-Image-Edit). Refine mode only; mutually exclusive "
        + "with Restore. Tips: shadows/reflections aren't under the brush — paint those too, or "
        + "they'll give the removal away. Left any artifacts behind? Turn Remove OFF and clean them "
        + "with the regular brush — Paint Mode on, in Refine mode, at a gentle Denoise (~0.6) — to "
        + "smooth them out without regenerating the whole area. Undo / Re-roll / Vary ×4 all work.";
    row1.appendChild(removeToggle);
    node._AngeloRemoveToggle = removeToggle;

    const fineUpscaleToggle = makeToggleButton("Xtra-Fine", () => {
        // Remove forces Xtra-Fine on and requires it — ignore clicks while
        // Remove is active so it can't be switched off underneath it.
        if (findWidget(node, "remove_mode")?.value) return;
        const w = findWidget(node, "fine_upscaling");
        if (!w) return;
        setWidget(w, !w.value);
        syncFineUpscaleToggle(node);
    });
    fineUpscaleToggle.title = "Xtra-Fine — refine the painted region at much higher effective resolution (ADetailer-style). The region is cropped, enlarged in pixel space to the MP target, re-encoded, refined, and composited back, so the model has room to render fine detail (faces, hands, eyes). Capped at Max scale.\n\nTip: pair it with Area Prompt — describe exactly what that region should be (e.g. \"detailed photorealistic face, sharp eyes\") for the strongest result.";
    row1.appendChild(fineUpscaleToggle);
    node._AngeloFineUpscaleToggle = fineUpscaleToggle;

    // Reference: a unified toggle + strength pair. The strength box only
    // exists while the toggle is lit — joined borders make them read as
    // one control.
    const refGroup = document.createElement("div");
    refGroup.style.cssText = "display:inline-flex; align-items:stretch; flex:0 0 auto;";
    const refineRefToggle = makeToggleButton("Reference", () => {
        const w = findWidget(node, "refine_reference");
        if (!w) return;
        const next = !w.value;
        setWidget(w, next);
        if (next) {
            // Switching ON must mean something — seed a sensible strength
            // if the box is still at zero.
            const sw = findWidget(node, "reference_strength");
            if (sw && !(Number(sw.value) > 0)) setWidget(sw, 0.8);
        }
        syncReferenceControls(node);
    });
    refineRefToggle.title = "Reference — anchor a refine to the current image so you can run high "
        + "Denoise without losing the subject. Refine mode, edit models only (FLUX 2 Klein / Qwen).\n\n"
        + "WHY: normally high denoise destroys identity along with the noise. With Reference ON the "
        + "edit model holds the person/scene from the image while the texture fully re-renders — "
        + "that's where the quality comes from. The restoration sweet spot is the box at 0.6–1.0 "
        + "with Denoise 0.7–1.0.\n\n"
        + "STRENGTH BOX (appears when ON): how hard it anchors. 1 = locked to the image; lower = "
        + "blends in more freedom to change (0.6 = 60% anchored). With Xtra-Fine ON the anchor is "
        + "the upscaled crop.\n\n"
        + "Turn it OFF when your Area Prompt is meant to CHANGE the region — anchoring fights the "
        + "change. (Values strictly between 0 and 1 run a second pass per step, slightly slower; "
        + "0 and 1 are free.)";
    refGroup.appendChild(refineRefToggle);
    const refineRefInput = makeNumberInput("", { min: 0, max: 1, step: 0.05, width: 46 }, (val) => {
        const w = findWidget(node, "reference_strength");
        if (!w) return;
        setWidget(w, val);
    });
    refineRefInput.title = "Reference strength: the anchored/free blend ratio (0.6 = 60% anchored). "
        + "Restoration sweet spot is usually 0.6–1.0.";
    refineRefInput.style.padding = "0";
    refineRefInput.style.gap = "0";
    if (refineRefInput._AngeloInput) {
        refineRefInput._AngeloInput.style.borderRadius = "0 3px 3px 0";
        refineRefInput._AngeloInput.style.borderLeft = "none";
        refineRefInput._AngeloInput.style.height = "100%";
        refineRefInput._AngeloInput.style.boxSizing = "border-box";
    }
    refGroup.appendChild(refineRefInput);
    row1.appendChild(refGroup);
    node._AngeloRefGroup = refGroup;
    node._AngeloRefineRefToggle = refineRefToggle;
    node._AngeloRefineRefInput = refineRefInput;

    row1.appendChild(makeSeparator());

    // Inpainting Mode dropdown — Refine / Insert V1 / Insert V2.
    // Refine = current behaviour (best for refining existing content)
    // Smart Inpaint = drag a rectangle; locks denoise=1.0 / Fine Upscale=ON /
    //                 Ctx Pad=0; adds reference_latents so an edit model
    //                 (Klein 9B) sees the scene through its edit branch.
    const inpaintModeWidget = findWidget(node, "inpainting_mode");
    const inpaintModeOptions = (inpaintModeWidget && inpaintModeWidget.options && inpaintModeWidget.options.values)
        ? inpaintModeWidget.options.values
        : ["Refine", "Smart Inpaint", "Smart Guided Inpaint", "Outpaint"];
    const inpaintModeSelect = makeDropdown("Inpaint",
        inpaintModeOptions,
        (val) => {
            const w = findWidget(node, "inpainting_mode");
            if (!w) return;
            setWidget(w, val);
            // Switching INTO Smart Inpaint: default feather to 15 (a soft
            // rectangle edge that blends the inserted content into the
            // surroundings — a useful default for this mode). It stays
            // user-adjustable afterwards — this only fires on the user's mode
            // pick, not on workflow load, so a saved feather value is
            // preserved across reloads.
            if (val === "Smart Inpaint") {
                const fw = findWidget(node, "feather_radius");
                if (fw) {
                    setWidget(fw, 15);
                    syncFeatherInput(node);
                }
            }
            // Mode change rewires the canvas interaction model — redraw
            // so the cursor / toolbar / overlays reflect it immediately.
            syncSmartInpaintLockedWidgets(node);
            redrawCanvasWithOverlays(node);
        }
    );
    inpaintModeSelect.title = "Inpainting Mode.\n\n"
        + "Refine — paint/click on the canvas to refine an existing region (faces, hands, textures). Partial-denoise from existing content.\n\n"
        + "Smart Inpaint — drag a rectangle on the canvas (click and hold one corner, release at the opposite). Adds NEW content in that region. Locks denoise=1.0 + Xtra-Fine=ON + Area Prompt=ON; injects reference_latents so an edit model's (FLUX 2 Klein 9B etc.) edit branch activates. Feather defaults to 15 (soft blend) but stays adjustable.\n\n"
        + "Smart Guided Inpaint — no painting or boxes. Pick a LOCATION from the dropdown above the Area Prompt (top left, center, bottom half, …); it's prepended to your prompt at run time (e.g. 'In the top left of the image, a red car') and the edit model places the content there across the whole image. Locks denoise=1.0 + Xtra-Fine=OFF + Area Prompt=ON; Feather and Persistent Mask disabled (no mask). Press 'Generate Guided Edit' to run. Coarse regions land most reliably.\n\n"
        + "Outpaint — extend the canvas. Use the arrow buttons, or click near an edge of the preview (a glowing band shows where the extension goes). Every result is shown in a review overlay first — Accept commits it as a NEW session base (history resets), Try again re-rolls it, Cancel costs nothing.";
    row1.appendChild(inpaintModeSelect);
    node._AngeloInpaintModeSelect = inpaintModeSelect;

    // ===== ROW 2: numeric values =====
    const clickRadiusInput = makeNumberInput("Click R", { min: 8, max: 1024, step: 4, width: 56 }, (val) => {
        const w = findWidget(node, "click_radius");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    clickRadiusInput.title = "Pixel-space radius of the refinement region for a single click. Also the brush size in paint mode.\n\nKeyboard (cursor over canvas): [ to shrink, ] to grow.";
    row2.appendChild(clickRadiusInput);
    node._AngeloClickRadiusInput = clickRadiusInput;

    const featherInput = makeNumberInput("Feather", { min: 0, max: 256, step: 4, width: 56 }, (val) => {
        const w = findWidget(node, "feather_radius");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    featherInput.title = "Pixel-space gaussian blur on the mask edge. Smooths the seam between refined region and preserved surroundings. Roughly click_radius / 4 is a good default.\n\nKeyboard (cursor over canvas): { (shift+[) to shrink, } (shift+]) to grow.";
    row2.appendChild(featherInput);
    node._AngeloFeatherInput = featherInput;

    const denoiseInput = makeNumberInput("Denoise", { min: 0.05, max: 1.0, step: 0.05, width: 56 }, (val) => {
        const w = findWidget(node, "denoise");
        if (!w) return;
        setWidget(w, val);
    });
    denoiseInput.title = "How much of the sampler trajectory to run on the refine. 0.3 = subtle touch-up, 0.6 = real redo, 0.9+ = essentially regenerate that region.\n\nKeyboard (cursor over canvas): , to decrease, . to increase.";
    row2.appendChild(denoiseInput);
    node._AngeloDenoiseInput = denoiseInput;

    const seedInput = makeNumberInput("Seed", { min: 0, max: 0xFFFFFFFFFFFFFFFF, step: 1, width: 120 }, (val) => {
        const w = findWidget(node, "seed");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    seedInput.title = "[Edit Mode] Seed for the refine pass. After each click the Seed Ctrl dropdown decides what happens — fixed (leave alone), randomize (new random), increment (+1), decrement (-1).";
    row2.appendChild(seedInput);
    node._AngeloSeedInput = seedInput;

    const seedCtrlSelect = makeDropdown("Ctrl",
        ["fixed", "increment", "decrement", "randomize"],
        (val) => {
            const w = findWidget(node, "seed_control");
            if (!w) return;
            setWidget(w, val);
        }
    );
    seedCtrlSelect.title = "[Edit Mode] After-click seed behaviour. Mirrors ComfyUI's standard seed control_after_generate dropdown.";
    row2.appendChild(seedCtrlSelect);
    node._AngeloSeedCtrlSelect = seedCtrlSelect;

    // Xtra-Fine value group (separator + MP/Max/Ctx Pad/Method). Only
    // shown while Xtra-Fine is effectively ON — the values mean nothing
    // otherwise, and hiding them halves row 2 for the common case. The
    // show/hide lives in syncFineUpscaleToggle.
    const fineSep = makeSeparator();
    row2.appendChild(fineSep);
    node._AngeloFineSep = fineSep;

    const mpInput = makeNumberInput("MP", { min: 0.1, max: 4.0, step: 0.1, width: 50 }, (val) => {
        const w = findWidget(node, "min_megapixels");
        if (!w) return;
        setWidget(w, val);
    });
    mpInput.title = "Xtra-Fine: target megapixels for the refine pass. Higher = bigger compute per click but sharper detail. Only used when Xtra-Fine is ON.";
    row2.appendChild(mpInput);
    node._AngeloMpInput = mpInput;

    const maxInput = makeNumberInput("Max", { min: 1.0, max: 16.0, step: 0.5, width: 50 }, (val) => {
        const w = findWidget(node, "max_upscale");
        if (!w) return;
        setWidget(w, val);
    });
    maxInput.title = "Xtra-Fine: hard cap on linear enlarge factor (8× = 64× area). Prevents pathological blow-up on tiny paints. Only used when Xtra-Fine is ON.";
    row2.appendChild(maxInput);
    node._AngeloMaxInput = maxInput;

    const ctxPadInput = makeNumberInput("Ctx Pad", { min: 0, max: 512, step: 8, width: 50 }, (val) => {
        const w = findWidget(node, "fine_context_pad");
        if (!w) return;
        setWidget(w, val);
    });
    ctxPadInput.title = "Xtra-Fine: pixel-space padding around the painted shape bbox before cropping. Gives the model surrounding context. Only used when Xtra-Fine is ON.";
    row2.appendChild(ctxPadInput);
    node._AngeloCtxPadInput = ctxPadInput;

    // Read the resize-method options from the underlying widget enum.
    const methodWidget = findWidget(node, "resize_method");
    const methodOptions = (methodWidget && methodWidget.options && methodWidget.options.values)
        ? methodWidget.options.values
        : ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"];
    const methodSelect = makeDropdown("Method",
        methodOptions,
        (val) => {
            const w = findWidget(node, "resize_method");
            if (!w) return;
            setWidget(w, val);
        }
    );
    methodSelect.title = "Xtra-Fine: pixel-space enlarge method. lanczos = sharpest with mild ringing; bilinear = smooth (great for skin/faces); bicubic = middle; nearest-exact = blocky preserves exact values; bislerp/area = niche. Only used when Xtra-Fine is ON.";
    row2.appendChild(methodSelect);
    node._AngeloMethodSelect = methodSelect;

    // ===== QUICK ACTIONS BAR: one-press magic buttons (Refine mode) =====
    const qaLabel = document.createElement("span");
    qaLabel.textContent = "✦ Quick Actions:";
    qaLabel.style.cssText = "font-size:11px; color:#bbb; padding:0 2px 0 4px; white-space:nowrap;";
    quickRow.appendChild(qaLabel);

    const quickFixBtn = makeActionButton("✨ Quick Photo Refine", () => triggerQuickPhotoRefine(node), "quickfix");
    quickFixBtn.title = "One-click photo refine — a true magic button with its own fixed recipe: "
        + "the WHOLE image runs through the Xtra-Fine pipeline (1.3MP working target — small "
        + "images get internally supersampled, refined, composited back) with the instruction "
        + "\"Keep the identity from image 1. make the image high quality.\", reference anchor "
        + "1.0, denoise 1.0, feather 0. Identity stays, texture re-renders. NO toolbar box "
        + "affects it — only the Seed applies (leave Ctrl on randomize and mash for variations); "
        + "Undo steps back through passes. Auto-tiles on canvases over ~1.6MP.\n\n"
        + "Needs an edit model (FLUX 2 Klein / Qwen-Image-Edit) + a wired CLIP — on non-edit models "
        + "the reference is ignored, so this REGENERATES the image instead (Undo brings it back). "
        + "Refine mode only.";
    quickRow.appendChild(quickFixBtn);
    node._AngeloQuickFixBtn = quickFixBtn;

    const qpWidget = findWidget(node, "quick_prompt_mode");
    const qpOptions = (qpWidget && qpWidget.options && qpWidget.options.values)
        ? qpWidget.options.values
        : ["Identity + Quality", "Restore Photo", "Identity + Colours", "Use Area Prompt"];
    const quickPromptSelect = makeDropdown("", qpOptions, (val) => {
        const w = findWidget(node, "quick_prompt_mode");
        if (w) setWidget(w, val);
    });
    quickPromptSelect.title = "Which instruction ✨ Quick Photo Refine runs with:\n"
        + "• Identity + Quality (default) — 'Keep the identity from image 1. make the image high quality.' "
        + "(Qwen-Image-Edit gets its tuned dust-and-scratches variant automatically.)\n"
        + "• Restore Photo — 'Keep the identity from image 1. restore the photo.' For damaged/old photos.\n"
        + "• Identity + Colours — adds a colour hold for images where the palette must not move.\n"
        + "• Use Area Prompt — your own text from the Area Prompt box drives the pass "
        + "(falls back to the default if the box is empty).";
    quickRow.appendChild(quickPromptSelect);
    node._AngeloQuickPromptSelect = quickPromptSelect;

    // Lite toggle — same Quick Photo Refine recipe at a gentler denoise.
    const liteToggle = makeToggleButton("Lite", () => {
        const w = findWidget(node, "quick_lite");
        if (!w) return;
        setWidget(w, !w.value);
        _syncToggle(node._AngeloLiteToggle, findWidget(node, "quick_lite")?.value, _TOGGLE_ON_COLORS.teal);
    });
    liteToggle.title = "Lite mode for ✨ Quick Photo Refine — the exact same recipe at a GENTLER "
        + "denoise (0.8 instead of 1.0). It re-renders less and stays closer to the input: a "
        + "lighter restore / cleanup rather than a full rebuild. Everything else (reference anchor, "
        + "prompt, target resolution, tiling, re-roll) is identical.";
    quickRow.appendChild(liteToggle);
    node._AngeloLiteToggle = liteToggle;
    _syncToggle(liteToggle, findWidget(node, "quick_lite")?.value, _TOGGLE_ON_COLORS.teal);

    // Separator after the Lite toggle — the prompt dropdown + Lite belong to
    // Quick Photo Refine, so the pipe groups them and sets them apart from
    // the upscale buttons that follow.
    quickRow.appendChild(makeSeparator());

    const upscaleBtn = makeActionButton("⬆ 2× Pixel", () => triggerPixelUpscale(node), "quickfix");
    upscaleBtn.title = "Pure pixel-space 2× upscale — lanczos, NO AI, deterministic. The image is "
        + "decoded, enlarged 2×, re-encoded, and committed immediately as the session's new base "
        + "(dimension change, so history resets — like loading a new photo). Nothing is invented "
        + "or re-rendered.\n\n"
        + "For AI enhancement afterwards, that's your next move: press ✨ Quick Photo Refine "
        + "(auto-tiles on the now-large canvas), or spot-edit with Xtra-Fine.";
    quickRow.appendChild(upscaleBtn);
    node._AngeloUpscaleBtn = upscaleBtn;

    const shrinkBtn = makeActionButton("⬇ Shrink", () => showShrinkPopup(node), "quickfix");
    shrinkBtn.title = "Downscale the image (no AI): pick a scale factor in the popup and Angelo "
        + "resamples it smaller with AREA averaging — the right method for shrinking (anti-aliased, "
        + "no ringing). The new dimensions (snapped to a multiple of 16) are shown before you "
        + "confirm. Committed as a fresh session base — dimension change, so history resets. "
        + "Refine mode only.";
    quickRow.appendChild(shrinkBtn);
    node._AngeloShrinkBtn = shrinkBtn;

    // ===== OUTPAINT ROW: direction + amount (Outpaint mode only) =====
    // Arrows extend the canvas in that direction; "All" pads every side
    // (zoom-out). The same action is available by clicking near an edge
    // of the preview — the row is the explicit/discoverable surface, the
    // canvas is the fast one. Every result goes through a review overlay
    // before anything commits.
    const opModeLabel = document.createElement("span");
    opModeLabel.textContent = "⛶ Outpaint:";
    opModeLabel.style.cssText = "font-size:11px; color:#bbb; padding:0 2px 0 4px; white-space:nowrap;";
    outpaintRow.appendChild(opModeLabel);

    const mkOutpaintBtn = (txt, dir, tip) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = txt;
        b.title = tip;
        b.style.cssText = "cursor:pointer; padding:3px 9px; font-size:13px; font-weight:bold; "
            + "border:1px solid rgba(120,190,235,0.7); border-radius:3px; "
            + "background:rgba(40,62,82,0.95); color:#d8eeff; line-height:1; "
            + "user-select:none; flex:0 0 auto;";
        b.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            triggerOutpaint(node, dir);
        });
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        return b;
    };
    outpaintRow.appendChild(mkOutpaintBtn("←", "left", "Extend the canvas to the LEFT by Amount px"));
    outpaintRow.appendChild(mkOutpaintBtn("↑", "up", "Extend the canvas UPWARD by Amount px"));
    outpaintRow.appendChild(mkOutpaintBtn("↓", "down", "Extend the canvas DOWNWARD by Amount px"));
    outpaintRow.appendChild(mkOutpaintBtn("→", "right", "Extend the canvas to the RIGHT by Amount px"));
    outpaintRow.appendChild(mkOutpaintBtn("⛶ All", "all", "Extend ALL four sides by Amount px (zoom-out)"));

    outpaintRow.appendChild(makeSeparator());

    const opAmountInput = makeNumberInput("Amount", { min: 16, max: 2048, step: 16, width: 56 }, (val) => {
        const w = findWidget(node, "outpaint_amount");
        if (w) setWidget(w, Math.round(val / 16) * 16);
    });
    opAmountInput.title = "How many pixels to extend the canvas by (snapped to /16 so any VAE lands on clean latent cells).";
    outpaintRow.appendChild(opAmountInput);
    node._AngeloOutpaintAmountInput = opAmountInput;

    const opOverlapInput = makeNumberInput("Overlap", { min: 0, max: 512, step: 8, width: 50 }, (val) => {
        const w = findWidget(node, "outpaint_overlap");
        if (w) setWidget(w, Math.round(val));
    });
    opOverlapInput.title = "Feathered band reaching this many pixels INTO the existing image — that band is redrawn so the seam blends instead of butting. 64 is a good default.";
    outpaintRow.appendChild(opOverlapInput);
    node._AngeloOutpaintOverlapInput = opOverlapInput;

    const opHint = document.createElement("span");
    opHint.textContent = "edge-click = extend · drag = protect · Shift = protect anywhere";
    opHint.style.cssText = "font-size:10px; color:#8aa; padding:0 4px; white-space:nowrap;";
    opHint.title = "Canvas gestures in Outpaint mode:\n"
        + "• Click near an edge — extend the canvas that way (a glowing band previews it).\n"
        + "• Drag in the interior — paint a PROTECT region (red). Protected pixels are "
        + "excluded from the Overlap band, so something near the frame edge (a car, a face) "
        + "stays exactly as-is while the rest of the seam still blends generously. "
        + "Brush size = Click R.\n"
        + "• Hold SHIFT — the protect brush wins everywhere, including the edge zone, so "
        + "you can start a stroke on something flush against the frame edge without "
        + "triggering an extension.";
    outpaintRow.appendChild(opHint);

    const opClearProtectBtn = document.createElement("button");
    opClearProtectBtn.type = "button";
    opClearProtectBtn.textContent = "✕ Protect";
    opClearProtectBtn.title = "Clear the painted protect region.";
    opClearProtectBtn.style.cssText = "cursor:pointer; padding:2px 8px; font-size:10px; "
        + "border:1px solid rgba(255,120,120,0.7); border-radius:3px; "
        + "background:rgba(80,40,40,0.95); color:#fbb; line-height:1.4; "
        + "user-select:none; flex:0 0 auto; display:none;";
    opClearProtectBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
    opClearProtectBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        node._AngeloProtect = [];
        const wpr = findWidget(node, "outpaint_protect");
        if (wpr) setWidget(wpr, "");
        syncOutpaintControls(node);
        redrawCanvasWithOverlays(node);
    });
    outpaintRow.appendChild(opClearProtectBtn);
    node._AngeloOutpaintClearProtectBtn = opClearProtectBtn;

    // ===== DETECT ROW: SAM 3 auto-segment (Refine + Smart Inpaint) =====
    const detLabel = document.createElement("span");
    detLabel.textContent = "🔍 Detect:";
    detLabel.style.cssText = "font-size:11px; color:#bbb; padding:0 2px 0 4px;";
    detectRow.appendChild(detLabel);

    const detText = document.createElement("input");
    detText.type = "text";
    detText.placeholder = "what to segment (e.g. the face)";
    detText.style.cssText = "flex:1 1 0; min-width:0; background:#1a1a1a; color:#ddd; "
        + "border:1px solid #555; border-radius:3px; padding:2px 6px; font-size:11px;";
    detText.title = "SAM 3 concept prompt — a noun phrase describing what to find. "
        + "Hit Detect to highlight every match; click the one you want to refine/inpaint it.";
    for (const ev of ["pointerdown", "mousedown", "keydown", "keyup"]) {
        detText.addEventListener(ev, (e) => e.stopPropagation());
    }
    detText.addEventListener("keydown", (e) => { if (e.key === "Enter") runDetect(node); });
    detectRow.appendChild(detText);
    node._AngeloDetectText = detText;

    const detConf = makeNumberInput("Conf", { min: 0.05, max: 0.95, step: 0.05, width: 48 }, () => {});
    detConf.title = "Detection confidence threshold. Lower (≈0.2–0.3) finds more / fainter matches.";
    if (detConf._AngeloInput) detConf._AngeloInput.value = "0.3";
    detectRow.appendChild(detConf);
    node._AngeloDetectConf = detConf;

    const detBtn = makeActionButton("Detect", () => runDetect(node), "neutral");
    detBtn.title = "Run SAM 3 on the current preview and highlight matches. Click a highlight to confirm; Esc / click empty space to dismiss.";
    detectRow.appendChild(detBtn);
    node._AngeloDetectBtn = detBtn;

    // Space + separator between Detect and the quick-presets dropdown.
    detectRow.appendChild(makeSeparator());

    // Quick-detect presets — selecting one runs SAM 3 immediately with
    // that concept (does NOT change the text box). Resets to the
    // placeholder after each pick so the same item can be re-run.
    const quickSel = document.createElement("select");
    quickSel.style.cssText = "font-size:11px; padding:2px 4px; border:1px solid #555; "
        + "border-radius:3px; background:#1a1a1a; color:#ddd; margin-left:4px; min-width:104px;";
    quickSel.title = "Quick-detect a common subject — runs SAM 3 immediately. Doesn't touch the text box.";
    // Placeholder first (index 0 — reset target), then grouped presets.
    const _ph = document.createElement("option");
    _ph.value = "Quick Detect…"; _ph.textContent = "Quick Detect…";
    quickSel.appendChild(_ph);
    const _QUICK_GROUPS = {
        "People": ["Person", "Face", "Eyes", "Mouth", "Teeth", "Nose", "Ears", "Hair", "Skin", "Beard"],
        "Body": ["Hands", "Fingers", "Arms", "Legs", "Feet", "Torso"],
        "Clothing": ["Clothing", "Dress", "Shirt", "Jacket", "Pants", "Shoes", "Hat", "Glasses", "Jewelry", "Bag"],
        "Animals": ["Animal", "Dog", "Cat", "Bird", "Horse"],
        "Scene": ["Background", "Sky", "Clouds", "Sun", "Moon", "Water", "Tree", "Grass", "Flowers", "Mountains", "Road"],
        "Objects": ["Building", "Window", "Door", "Car", "Furniture", "Food", "Bottle", "Phone", "Text", "Logo"],
    };
    for (const [group, items] of Object.entries(_QUICK_GROUPS)) {
        const og = document.createElement("optgroup");
        og.label = group;
        for (const it of items) {
            const opt = document.createElement("option");
            opt.value = it;
            opt.textContent = it;
            og.appendChild(opt);
        }
        quickSel.appendChild(og);
    }
    quickSel.addEventListener("change", () => {
        const v = quickSel.value;
        quickSel.selectedIndex = 0;            // reset to "Quick Detect…"
        if (v && v !== "Quick Detect…") runDetect(node, v.toLowerCase());
    });
    quickSel.addEventListener("pointerdown", (e) => e.stopPropagation());
    quickSel.addEventListener("mousedown", (e) => e.stopPropagation());
    detectRow.appendChild(quickSel);
    node._AngeloDetectQuick = quickSel;

    // Mask grow / shrink — nudges ALL detected masks in/out together, 2px
    // at a time, so a tight SAM silhouette can be loosened (or a loose one
    // tightened) before committing. Pure-frontend: offsets the polygons /
    // bbox the JS already holds; the backend rasterises whatever it gets.
    // Lives in the FLOATING DETECT PANEL (appended there below), not this
    // always-visible row — it only means anything while candidates are up,
    // and the detect row is nowrap-tight on narrow nodes.
    const maskRow = document.createElement("div");
    maskRow.style.cssText = "display:flex; align-items:center; justify-content:center; gap:4px;";
    const maskLabel = document.createElement("span");
    maskLabel.textContent = "Mask:";
    maskLabel.style.cssText = "font-size:11px; color:#bbb; padding:0 1px; white-space:nowrap;";
    maskRow.appendChild(maskLabel);

    const mkGrowBtn = (txt, delta, tip) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = txt;
        b.title = tip;
        b.style.cssText = "cursor:pointer; padding:2px 7px; font-size:13px; font-weight:bold; "
            + "border:1px solid #555; border-radius:3px; background:#2a2a2a; color:#ddd; "
            + "line-height:1; user-select:none; flex:0 0 auto;";
        b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); adjustMaskGrow(node, delta); });
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        return b;
    };
    maskRow.appendChild(mkGrowBtn("−", -2, "Shrink all detected masks by 2px"));
    const growReadout = document.createElement("span");
    growReadout.textContent = "0px";
    growReadout.style.cssText = "font-size:11px; color:#9cf; min-width:34px; text-align:center; white-space:nowrap;";
    maskRow.appendChild(growReadout);
    node._AngeloMaskGrowReadout = growReadout;
    maskRow.appendChild(mkGrowBtn("+", 2, "Grow all detected masks by 2px"));

    // ===== MODE ROW: the master Sampler/Edit switch, centred up top =====
    const modeWidget = findWidget(node, "mode");
    const modeOptions = (modeWidget && modeWidget.options && modeWidget.options.values)
        ? modeWidget.options.values
        : ["Sampler Mode", "Edit Mode"];
    const modeSelect = makeDropdown("Mode",
        modeOptions,
        (val) => {
            const w = findWidget(node, "mode");
            if (!w) return;
            setWidget(w, val);   // fires the wrapped mode callback (lock + grey sync)
        }
    );
    modeSelect.title = "Sampler Mode = generate a fresh base latent from the inputs (acts like a KSampler). Edit Mode = click/drag the preview to refine or inpaint the cached latent. Switching to Edit Mode auto-locks the sampler seed to the value that produced the base; switching back restores your previous seed control (and pre-rolls a fresh seed if it was randomize, so the next Queue is a new base). Right-click the preview for quick actions: switch mode, generate new base, regenerate same base.";
    modeRow.appendChild(modeSelect);
    node._AngeloModeSelect = modeSelect;

    // Fullscreen toggle — pinned top-left of the Mode row (mirrors the
    // floating detect panel on the right, so it never disturbs the centred
    // Mode dropdown). Pops the WHOLE editor (toolbar + canvas) into a
    // viewport-filling overlay for a much bigger canvas; Esc or the button
    // returns it. Works in both Sampler and Edit mode.
    const fullscreenBtn = document.createElement("button");
    fullscreenBtn.type = "button";
    fullscreenBtn.textContent = "⛶ Fullscreen";
    fullscreenBtn.style.cssText = "position:absolute; left:6px; top:4px; z-index:6; cursor:pointer; "
        + "padding:3px 9px; font-size:11px; font-weight:bold; border-radius:3px; "
        + "border:1px solid #555; background:#2a2a2a; color:#ccc; user-select:none; line-height:15px;";
    for (const ev of ["pointerdown", "mousedown"]) {
        fullscreenBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    fullscreenBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleAngeloFullscreen(node);
    });
    modeRow.appendChild(fullscreenBtn);
    node._AngeloFullscreenBtn = fullscreenBtn;
    syncFullscreenButton(node);

    // ▶ Queue — the in-node twin of ComfyUI's Queue button. Same code path
    // (app.queuePrompt via the queuePrompt() helper), so the persistent-mask
    // queue hook and everything else behave identically — but you never have
    // to leave the node (or fullscreen) to press it. Works in both modes:
    // Sampler Mode = generate the base; Edit Mode = re-execute the graph
    // (e.g. Persistent Mask re-runs, upstream changes).
    const queueBtn = document.createElement("button");
    queueBtn.type = "button";
    queueBtn.textContent = "▶ Queue";
    queueBtn.title = "Queue the workflow — identical to ComfyUI's main Queue button, right here in "
        + "the node (works in fullscreen too).\n\n"
        + "Sampler Mode: generates a fresh base image.\n"
        + "Edit Mode: re-executes the graph — with Persistent Mask ON each press re-runs the held "
        + "mask on the latest result with a fresh seed (the gradual-morph loop), and it's also the "
        + "button to press after changing something upstream.";
    queueBtn.style.cssText = "cursor:pointer; padding:3px 12px; font-size:11px; font-weight:bold; "
        + "border:1px solid rgba(140, 220, 170, 0.9); border-radius:3px; "
        + "background:rgba(30, 120, 80, 0.95); color:#fff; user-select:none; line-height:15px; "
        + "margin-left:6px;";
    for (const ev of ["pointerdown", "mousedown"]) {
        queueBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    queueBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        queuePrompt();
    });
    modeRow.appendChild(queueBtn);
    node._AngeloQueueBtn = queueBtn;

    // Floating detect-mode panel — pinned top-right of the Mode row, shown
    // only while candidates are active. Holds the red Cancel button + an
    // opacity slider for the highlight overlay (drag down to peek at the
    // edges of a just-generated region). Styled as one neat cluster.
    node._AngeloDetOpacity = 1.0;
    const detectPanel = document.createElement("div");
    detectPanel.style.cssText = "position:absolute; right:6px; top:4px; z-index:6; display:none; "
        + "flex-direction:column; align-items:stretch; gap:4px; padding:5px 6px; "
        + "background:#000; border:1px solid #555; border-radius:5px;";

    const cancelDetectBtn = document.createElement("button");
    cancelDetectBtn.type = "button";
    cancelDetectBtn.textContent = "✕ Cancel Detect";
    cancelDetectBtn.title = "Leave detect mode (you can also press Esc or Space). "
        + "The highlighted candidates stay up so you can edit each one until you cancel.";
    cancelDetectBtn.style.cssText = "font-size:11px; font-weight:bold; padding:3px 10px; "
        + "border:1px solid #e66; border-radius:3px; background:rgba(200,40,40,0.95); "
        + "color:#fff; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        cancelDetectBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    cancelDetectBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearDetections(node);
    });

    const opRow = document.createElement("div");
    opRow.style.cssText = "display:flex; align-items:center; gap:6px; font-size:10px; color:#ccd;";
    const opLabel = document.createElement("span");
    opLabel.textContent = "Highlight";
    const opSlider = document.createElement("input");
    opSlider.type = "range";
    opSlider.min = "0"; opSlider.max = "1"; opSlider.step = "0.05"; opSlider.value = "1";
    opSlider.style.cssText = "flex:1 1 auto; width:96px; cursor:pointer;";
    opSlider.title = "Selection-highlight opacity — drag down to peek at the edges of what you just generated. Candidates stay clickable.";
    for (const ev of ["pointerdown", "mousedown"]) {
        opSlider.addEventListener(ev, (e) => e.stopPropagation());
    }
    opSlider.addEventListener("input", () => {
        node._AngeloDetOpacity = parseFloat(opSlider.value);
        redrawCanvasWithOverlays(node);
    });
    opRow.appendChild(opLabel);
    opRow.appendChild(opSlider);

    // Touch-up brush hint (the brush is modifier-driven, so flag it here).
    const brushHint = document.createElement("div");
    brushHint.textContent = "Refine: Shift-drag = +mask · Alt-drag = −mask";
    brushHint.style.cssText = "font-size:9px; color:#9aa; white-space:nowrap; text-align:center;";
    brushHint.title = "In Refine, hold Shift and drag on the preview to grow the mask you start over, "
        + "or Alt-drag to carve it back (holes allowed). Brush size = Click R. Then click the candidate to apply.";

    // Fix All — automatically work through every remaining candidate.
    // Confirms one, waits for its run to land (onExecuted), confirms the
    // next; the green edited-tracking shows progress on the canvas while
    // the button doubles as a Stop control with a live count.
    const fixAllBtn = document.createElement("button");
    fixAllBtn.type = "button";
    fixAllBtn.textContent = "⚡ Fix All";
    fixAllBtn.title = "Automatically edit every remaining (non-green) candidate in turn with the "
        + "current settings — Area Prompt, Xtra-Fine, seed control, all apply per candidate. "
        + "Each one is a separate history entry, so individual Undo still works. "
        + "Click again to stop after the in-flight candidate.";
    fixAllBtn.style.cssText = "font-size:11px; font-weight:bold; padding:3px 10px; "
        + "border:1px solid #4a7; border-radius:3px; background:rgba(30,120,80,0.95); "
        + "color:#fff; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        fixAllBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    fixAllBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleFixAll(node);
    });

    detectPanel.appendChild(cancelDetectBtn);
    detectPanel.appendChild(fixAllBtn);
    detectPanel.appendChild(opRow);
    detectPanel.appendChild(maskRow);
    detectPanel.appendChild(brushHint);
    modeRow.appendChild(detectPanel);
    node._AngeloCancelDetectBtn = cancelDetectBtn;
    node._AngeloFixAllBtn = fixAllBtn;
    node._AngeloDetectPanel = detectPanel;
    node._AngeloDetOpacitySlider = opSlider;

    // ===== ROW 3: shared generation config (always active) =====
    const stepsInput = makeNumberInput("Steps", { min: 1, max: 100, step: 1, width: 48 }, (val) => {
        const w = findWidget(node, "steps");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    stepsInput.title = "Sampler step count for both Sampler Mode and refines. Match the model — FLUX 2 Klein distilled = 4.";
    row3.appendChild(stepsInput);
    node._AngeloStepsInput = stepsInput;

    const cfgInput = makeNumberInput("CFG", { min: 0.0, max: 30.0, step: 0.1, width: 48 }, (val) => {
        const w = findWidget(node, "cfg");
        if (!w) return;
        setWidget(w, val);
    });
    cfgInput.title = "Classifier-free guidance scale. FLUX 2 Klein distilled uses CFG=1 (no negative branch).";
    row3.appendChild(cfgInput);
    node._AngeloCfgInput = cfgInput;

    const samplerWidget = findWidget(node, "sampler_name");
    const samplerOptions = (samplerWidget && samplerWidget.options && samplerWidget.options.values)
        ? samplerWidget.options.values
        : ["euler"];
    const samplerSelect = makeDropdown("Sampler",
        samplerOptions,
        (val) => {
            const w = findWidget(node, "sampler_name");
            if (!w) return;
            setWidget(w, val);
        }
    );
    samplerSelect.title = "Sampling algorithm (shared by Sampler Mode + refines).";
    row3.appendChild(samplerSelect);
    node._AngeloSamplerSelect = samplerSelect;

    const schedulerWidget = findWidget(node, "scheduler");
    const schedulerOptions = (schedulerWidget && schedulerWidget.options && schedulerWidget.options.values)
        ? schedulerWidget.options.values
        : ["simple"];
    const schedulerSelect = makeDropdown("Sched",
        schedulerOptions,
        (val) => {
            const w = findWidget(node, "scheduler");
            if (!w) return;
            setWidget(w, val);
        }
    );
    schedulerSelect.title = "Noise schedule (shared by Sampler Mode + refines).";
    row3.appendChild(schedulerSelect);
    node._AngeloSchedulerSelect = schedulerSelect;

    // Load Image — bring an external photo in as the base to edit. Always
    // active (both modes), which is exactly the Mode row's semantics — so
    // it lives there, to the left of the Mode dropdown, keeping row 3 as
    // pure sampler config and using the Mode row's otherwise-empty space.
    const loadImgBtn = makeActionButton("🖼 Load Image", () => triggerLoadImage(node), "neutral");
    loadImgBtn.title = "Load an external image as the base to edit / refine. "
        + "You'll be asked to keep its resolution or resize to a target "
        + "megapixel (both rounded to a /16 multiple). The image becomes the "
        + "base — Reset and Undo return to it. While loaded, the latent input "
        + "is ignored (hit Unload to go back to it). No Empty Latent needed.";
    modeRow.insertBefore(loadImgBtn, modeSelect);
    node._AngeloLoadImageBtn = loadImgBtn;

    const unloadImgBtn = makeActionButton("✕ Unload", () => unloadImage(node), "neutral");
    unloadImgBtn.title = "Clear the loaded image and return to the wired latent input "
        + "as the base. Shown only while an image is loaded.";
    unloadImgBtn.style.display = "none";
    modeRow.insertBefore(unloadImgBtn, modeSelect);
    node._AngeloUnloadImageBtn = unloadImgBtn;

    // ===== ROW 4: Sampler-Mode seed group (greyed in Edit Mode) =====
    const samplerSeedInput = makeNumberInput("Smpl Seed", { min: 0, max: 0xFFFFFFFFFFFFFFFF, step: 1, width: 120 }, (val) => {
        const w = findWidget(node, "sampler_seed");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    samplerSeedInput.title = "[Sampler Mode] Seed for the base generation. After each run the Sampler Ctrl dropdown decides what happens to it.";
    row4.appendChild(samplerSeedInput);
    node._AngeloSamplerSeedInput = samplerSeedInput;

    const samplerSeedCtrlSelect = makeDropdown("Smpl Ctrl",
        ["fixed", "increment", "decrement", "randomize"],
        (val) => {
            const w = findWidget(node, "sampler_seed_control");
            if (!w) return;
            setWidget(w, val);
        }
    );
    samplerSeedCtrlSelect.title = "[Sampler Mode] After-generate seed behaviour for the base. Auto-forced to 'fixed' when you switch to Edit Mode so re-queues don't regenerate the base.";
    row4.appendChild(samplerSeedCtrlSelect);
    node._AngeloSamplerSeedCtrlSelect = samplerSeedCtrlSelect;

    // min matches the Python widget (0.05) — a 0.0 here used to pass JS
    // clamping but fail ComfyUI's server-side validation on the next Queue.
    const samplerDenoiseInput = makeNumberInput("Smpl Denoise", { min: 0.05, max: 1.0, step: 0.05, width: 56 }, (val) => {
        const w = findWidget(node, "sampler_denoise");
        if (!w) return;
        setWidget(w, val);
    });
    samplerDenoiseInput.title = "[Sampler Mode] Denoise for the base generation. 1.0 = full generation from the incoming (usually empty) latent.";
    row4.appendChild(samplerDenoiseInput);
    node._AngeloSamplerDenoiseInput = samplerDenoiseInput;

    container.appendChild(toggleBarWrap);

    // --- Area Prompt input — sits BETWEEN the toolbar and the canvas,
    //     and only shows when Area Prompt is ON (or Smart Inpaint forces
    //     it on). One textarea with a Pos/Neg toggle deciding which
    //     underlying widget it edits (area_text_positive /
    //     area_text_negative). Hiding it never clears the text — that
    //     lives in the widgets, and syncAreaPromptBox reloads it. ---
    attachAreaPromptBox(node, container);

    // Canvas + placeholder live in their own relative-positioned wrap
    // so the absolutely-positioned placeholder overlays ONLY the canvas.
    // The wrap flex-grows to fill whatever vertical space is left after
    // the toolbar + area-prompt box, and centres the canvas inside it.
    // The canvas's DISPLAY size is computed in JS (fitCanvasDisplaySize)
    // to fit this wrap while preserving aspect ratio — letterboxed by
    // the empty space around it, NOT by object-fit (which would break
    // the click-coordinate mapping that reads canvas.getBoundingClientRect).
    const canvasWrap = document.createElement("div");
    canvasWrap.style.position = "relative";
    canvasWrap.style.width = "100%";
    canvasWrap.style.flex = "1 1 auto";
    canvasWrap.style.minHeight = "0";
    canvasWrap.style.display = "flex";
    canvasWrap.style.alignItems = "center";
    canvasWrap.style.justifyContent = "center";
    canvasWrap.style.overflow = "hidden";

    // The canvas is ABSOLUTELY positioned inside the wrap — applyView()
    // sets its width/height (= fit size × zoom) and left/top (centre +
    // pan). This lets zoom>1 overflow the wrap (clipped by overflow:
    // hidden) while click/paint/overlay mapping stays correct, because
    // they all read canvas.getBoundingClientRect() which reflects the
    // live size+position. No max-width/height (would block zoom>1).
    const canvas = document.createElement("canvas");
    canvas.style.display = "block";
    canvas.style.position = "absolute";
    canvas.style.cursor = "crosshair";
    canvas.width = 512;
    canvas.height = 512;
    canvasWrap.appendChild(canvas);

    // Placeholder text shown until the first image arrives
    const placeholder = document.createElement("div");
    placeholder.textContent = "Queue the workflow to generate a preview.\nClick a region in the preview to refine it.";
    placeholder.style.position = "absolute";
    placeholder.style.inset = "0";
    placeholder.style.display = "flex";
    placeholder.style.alignItems = "center";
    placeholder.style.justifyContent = "center";
    placeholder.style.textAlign = "center";
    placeholder.style.color = "#888";
    placeholder.style.padding = "20px";
    placeholder.style.whiteSpace = "pre-line";
    placeholder.style.pointerEvents = "none";
    placeholder.style.fontSize = "12px";
    canvasWrap.appendChild(placeholder);

    // Corner minimap navigator — shown only when zoomed in (zoom > 1).
    const minimap = document.createElement("canvas");
    minimap.style.position = "absolute";
    minimap.style.right = "6px";
    minimap.style.bottom = "6px";
    minimap.style.border = "1px solid rgba(255,255,255,0.35)";
    minimap.style.borderRadius = "2px";
    minimap.style.background = "rgba(0,0,0,0.4)";
    minimap.style.pointerEvents = "none";
    minimap.style.display = "none";
    minimap.style.zIndex = "5";
    canvasWrap.appendChild(minimap);
    node._AngeloMinimap = minimap;

    // Persistent in-app notice bar — overlays the top of the preview for
    // actionable messages (e.g. "SAM 3 Detect isn't installed — run the
    // installer"). Stays until dismissed (✕) or a detect succeeds; this is
    // NOT a transient toast, because the message needs reading + acting on.
    const notice = document.createElement("div");
    notice.style.cssText = "position:absolute; left:0; right:0; top:0; z-index:7; display:none; "
        + "padding:8px 28px 8px 10px; background:rgba(150,40,40,0.96); color:#fff; "
        + "font:12px/1.45 Arial,sans-serif; white-space:pre-line; "
        + "border-bottom:1px solid rgba(255,255,255,0.25);";
    const noticeText = document.createElement("span");
    notice.appendChild(noticeText);
    const noticeClose = document.createElement("button");
    noticeClose.type = "button";
    noticeClose.textContent = "✕";
    noticeClose.title = "Dismiss";
    noticeClose.style.cssText = "position:absolute; right:5px; top:5px; background:transparent; "
        + "border:none; color:#fff; font-size:14px; line-height:1; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        noticeClose.addEventListener(ev, (e) => e.stopPropagation());
    }
    noticeClose.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); notice.style.display = "none"; });
    notice.appendChild(noticeClose);
    canvasWrap.appendChild(notice);
    node._AngeloNotice = notice;
    node._AngeloNoticeText = noticeText;

    // Loading overlay — shown while a Detect request is in flight. SAM 3's
    // FIRST detect has to build + cache the model (several seconds), during
    // which the canvas would otherwise look frozen before the first outline
    // appears. Auto-dismisses when the detect resolves; the ✕ is a manual
    // escape hatch so a hung/crashed request can never trap the overlay up.
    if (!document.getElementById("angelo-spin-style")) {
        const s = document.createElement("style");
        s.id = "angelo-spin-style";
        s.textContent = "@keyframes angelo-spin{to{transform:rotate(360deg)}}";
        document.head.appendChild(s);
    }
    const loading = document.createElement("div");
    loading.style.cssText = "position:absolute; inset:0; z-index:8; display:none; "
        + "align-items:center; justify-content:center; background:rgba(0,0,0,0.55);";
    const loadingBox = document.createElement("div");
    loadingBox.style.cssText = "position:relative; display:flex; align-items:center; gap:11px; "
        + "padding:15px 36px 15px 18px; background:rgba(28,28,28,0.97); color:#fff; "
        + "font:13px/1.4 Arial,sans-serif; border-radius:8px; "
        + "border:1px solid rgba(255,255,255,0.18); box-shadow:0 4px 16px rgba(0,0,0,0.55);";
    const spinner = document.createElement("div");
    spinner.style.cssText = "flex:0 0 auto; width:16px; height:16px; "
        + "border:2px solid rgba(255,255,255,0.25); border-top-color:#fff; "
        + "border-radius:50%; animation:angelo-spin 0.8s linear infinite;";
    const loadingText = document.createElement("span");
    loadingText.textContent = "Loading SAM 3…";
    loadingBox.appendChild(spinner);
    loadingBox.appendChild(loadingText);
    const loadingClose = document.createElement("button");
    loadingClose.type = "button";
    loadingClose.textContent = "✕";
    loadingClose.title = "Dismiss";
    loadingClose.style.cssText = "position:absolute; right:6px; top:4px; background:transparent; "
        + "border:none; color:#fff; font-size:13px; line-height:1; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        loadingClose.addEventListener(ev, (e) => e.stopPropagation());
    }
    loadingClose.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); loading.style.display = "none"; });
    loadingBox.appendChild(loadingClose);
    loading.appendChild(loadingBox);
    canvasWrap.appendChild(loading);
    node._AngeloLoading = loading;
    node._AngeloLoadingText = loadingText;

    // Vary ×4 chooser — a 2×2 grid of candidate previews over the canvas.
    // Click one to commit it (replaces the last attempt); ✕ or Esc keeps
    // the current result. Populated by showVaryChooser when a vary run's
    // Angelo_vary_previews message arrives.
    const varyOverlay = document.createElement("div");
    varyOverlay.style.cssText = "position:absolute; inset:0; z-index:9; display:none; "
        + "flex-direction:column; gap:6px; padding:8px; background:rgba(0,0,0,0.8);";
    const varyHeader = document.createElement("div");
    varyHeader.style.cssText = "display:flex; align-items:center; justify-content:space-between; "
        + "color:#ddd; font:bold 12px Arial,sans-serif;";
    const varyTitle = document.createElement("span");
    varyTitle.textContent = "Pick a variation — click one (✕ keeps the current result)";
    const varyClose = document.createElement("button");
    varyClose.type = "button";
    varyClose.textContent = "✕";
    varyClose.title = "Close without picking — keeps the current result";
    varyClose.style.cssText = "background:transparent; border:1px solid #666; border-radius:3px; "
        + "color:#fff; font-size:13px; line-height:1; padding:3px 8px; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        varyClose.addEventListener(ev, (e) => e.stopPropagation());
    }
    varyClose.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        hideVaryChooser(node);
    });
    varyHeader.appendChild(varyTitle);
    varyHeader.appendChild(varyClose);
    const varyGrid = document.createElement("div");
    varyGrid.style.cssText = "flex:1 1 auto; min-height:0; display:grid; "
        + "grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:6px;";
    varyOverlay.appendChild(varyHeader);
    varyOverlay.appendChild(varyGrid);
    // Swallow canvas-bound pointer events so a chooser click can't paint.
    for (const ev of ["pointerdown", "mousedown", "click", "wheel"]) {
        varyOverlay.addEventListener(ev, (e) => e.stopPropagation());
    }
    canvasWrap.appendChild(varyOverlay);
    node._AngeloVaryOverlay = varyOverlay;
    node._AngeloVaryGrid = varyGrid;

    // Outpaint review — the extended canvas shown full-size with three
    // choices. NOTHING commits until Accept: Try again re-rolls the same
    // extension with a fresh seed, Cancel/Esc walks away free. Accepting
    // installs the new canvas as a fresh session base (history resets —
    // stated right on the button so it's never a surprise).
    const opOverlay = document.createElement("div");
    opOverlay.style.cssText = "position:absolute; inset:0; z-index:10; display:none; "
        + "flex-direction:column; gap:6px; padding:8px; background:rgba(0,0,0,0.85);";
    const opHeader = document.createElement("div");
    opHeader.style.cssText = "display:flex; align-items:center; justify-content:space-between; "
        + "color:#ddd; font:bold 12px Arial,sans-serif;";
    const opTitle = document.createElement("span");
    opTitle.textContent = "Outpaint preview — keep it?";
    const opEsc = document.createElement("span");
    opEsc.textContent = "Esc = cancel";
    opEsc.style.cssText = "font-weight:normal; font-size:10px; color:#8aa;";
    opHeader.appendChild(opTitle);
    opHeader.appendChild(opEsc);
    node._AngeloOutpaintTitle = opTitle;
    const opImgWrap = document.createElement("div");
    opImgWrap.style.cssText = "flex:1 1 auto; min-height:0; display:flex; "
        + "align-items:center; justify-content:center;";
    const opImg = document.createElement("img");
    opImg.style.cssText = "max-width:100%; max-height:100%; object-fit:contain; display:block; "
        + "border:1px solid #444; border-radius:3px;";
    opImg.draggable = false;
    opImgWrap.appendChild(opImg);
    const opBtnRow = document.createElement("div");
    opBtnRow.style.cssText = "display:flex; justify-content:center; gap:8px;";
    const mkOpReviewBtn = (txt, css) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = txt;
        b.style.cssText = "font-size:12px; font-weight:bold; padding:5px 14px; border-radius:4px; "
            + "cursor:pointer; " + css;
        for (const ev of ["pointerdown", "mousedown"]) {
            b.addEventListener(ev, (e) => e.stopPropagation());
        }
        return b;
    };
    const opAcceptBtn = mkOpReviewBtn("✓ Accept (new base — history resets)",
        "border:1px solid #4a7; background:rgba(30,120,80,0.95); color:#fff;");
    opAcceptBtn.title = "Commit the extended canvas as the session's new base image. "
        + "Like loading a new photo: Undo history resets, and Reset / Restore / the \\ compare "
        + "key all anchor to this new canvas.";
    const opRetryBtn = mkOpReviewBtn("🎲 Try again",
        "border:1px solid rgba(170,130,220,0.9); background:rgba(58,50,72,0.95); color:#ecdcff;");
    opRetryBtn.title = "Re-roll the same extension with a fresh seed. Nothing has been committed.";
    const opCancelBtn = mkOpReviewBtn("✕ Cancel",
        "border:1px solid #555; background:#2a2a2a; color:#ccc;");
    opCancelBtn.title = "Walk away — the canvas stays exactly as it was. Nothing was committed.";
    opAcceptBtn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        hideOutpaintReview(node);
        triggerOutpaintAccept(node);
    });
    opRetryBtn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        hideOutpaintReview(node);
        triggerOutpaintRetry(node);
    });
    opCancelBtn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        hideOutpaintReview(node);
    });
    opBtnRow.appendChild(opAcceptBtn);
    opBtnRow.appendChild(opRetryBtn);
    opBtnRow.appendChild(opCancelBtn);
    opOverlay.appendChild(opHeader);
    opOverlay.appendChild(opImgWrap);
    opOverlay.appendChild(opBtnRow);
    for (const ev of ["pointerdown", "mousedown", "click", "wheel"]) {
        opOverlay.addEventListener(ev, (e) => e.stopPropagation());
    }
    canvasWrap.appendChild(opOverlay);
    node._AngeloOutpaintOverlay = opOverlay;
    node._AngeloOutpaintImg = opImg;

    // Per-node view state (zoom/pan). zoom 1 = fit; pan in CSS px.
    node._AngeloZoom = 1;
    node._AngeloPanX = 0;
    node._AngeloPanY = 0;
    node._AngeloBaseW = 0;
    node._AngeloBaseH = 0;

    container.appendChild(canvasWrap);

    // --- Zoom / pan (view layer, independent of the refine pipeline) ---
    //   • Wheel       → zoom toward the cursor (clamped fit–8×; there's
    //                   no use for an image smaller than the panel, so
    //                   zooming out stops at fit — #29).
    //   • Middle-drag → pan.
    //   • Double-middle-click, or 'F' (cursor over node) → reset to fit.
    // While zoomed/panned the auto-fit (fitCanvasDisplaySize) is suppressed
    // so it never stomps the manual view; reset / new image restore fit.

    canvasWrap.addEventListener("wheel", (event) => {
        if (!node._AngeloImg || !node._AngeloBaseW) return;
        event.preventDefault();
        event.stopPropagation();   // don't also zoom the ComfyUI graph
        const wrapW = canvasWrap.clientWidth, wrapH = canvasWrap.clientHeight;
        const wrapRect = canvasWrap.getBoundingClientRect();
        // Convert cursor from viewport pixels to wrap-layout pixels.
        // ComfyUI applies a CSS transform: scale() on the graph container
        // when the user zooms the graph; getBoundingClientRect() reflects
        // that scale (visual pixels) while clientWidth/Height stay at the
        // unscaled layout size. Without dividing by the ratio the cursor
        // anchor drifts sideways under graph zoom and the image walks off
        // on each wheel tick. Thanks to @KursatAs (#23) for the diagnosis.
        const graphScaleX = wrapRect.width > 0 ? wrapRect.width / wrapW : 1;
        const graphScaleY = wrapRect.height > 0 ? wrapRect.height / wrapH : 1;
        const cx = (event.clientX - wrapRect.left) / graphScaleX;
        const cy = (event.clientY - wrapRect.top) / graphScaleY;
        const oldZoom = node._AngeloZoom || 1;
        const factor = event.deltaY < 0 ? 1.15 : (1 / 1.15);
        const newZoom = Math.max(1, Math.min(8, oldZoom * factor));
        if (newZoom === oldZoom) return;
        // Wheeling back through ~1× snaps to a clean fit (zoom=1, no pan).
        // Without this, float drift leaves zoom at e.g. 1.0000002, which
        // keeps the minimap up and the auto-fit suppressed at what looks
        // like fit.
        if (Math.abs(newZoom - 1) < 0.01) {
            resetView(node);
            redrawCanvasWithOverlays(node);
            return;
        }
        const baseW = node._AngeloBaseW, baseH = node._AngeloBaseH;
        const oldW = baseW * oldZoom, oldH = baseH * oldZoom;
        const oldLeft = (wrapW - oldW) / 2 + (node._AngeloPanX || 0);
        const oldTop = (wrapH - oldH) / 2 + (node._AngeloPanY || 0);
        // Normalised image point currently under the cursor.
        const nx = (cx - oldLeft) / oldW;
        const ny = (cy - oldTop) / oldH;
        const newW = baseW * newZoom, newH = baseH * newZoom;
        // Solve pan so that same normalised point stays under the cursor.
        node._AngeloZoom = newZoom;
        node._AngeloPanX = cx - nx * newW - (wrapW - newW) / 2;
        node._AngeloPanY = cy - ny * newH - (wrapH - newH) / 2;
        applyView(node);
        redrawCanvasWithOverlays(node);
    }, { passive: false });

    // Suppress the Windows middle-click autoscroll cursor.
    canvasWrap.addEventListener("mousedown", (event) => {
        if (event.button === 1) event.preventDefault();
    });

    canvasWrap.addEventListener("pointerdown", (event) => {
        if (event.button !== 1) return;   // middle button = pan / reset
        event.preventDefault();
        event.stopPropagation();   // don't let litegraph treat it as a node drag
        const now = performance.now();
        if (node._AngeloLastMiddleDown && (now - node._AngeloLastMiddleDown) < 350) {
            // Double middle-click → reset to fit.
            node._AngeloLastMiddleDown = 0;
            node._AngeloPanning = null;
            resetView(node);
            redrawCanvasWithOverlays(node);
            return;
        }
        node._AngeloLastMiddleDown = now;
        try { canvasWrap.setPointerCapture(event.pointerId); } catch (e) { /* noop */ }
        // clientX/Y deltas are VISUAL px, but _AngeloPanX/Y are LAYOUT px —
        // when the ComfyUI graph is zoomed the two differ by the graph scale,
        // making the pan run fast/slow (#23's wheel fix, applied to pan).
        // Capture the ratio once at drag start.
        const wrapRect = canvasWrap.getBoundingClientRect();
        node._AngeloPanning = {
            startX: event.clientX, startY: event.clientY,
            startPanX: node._AngeloPanX || 0, startPanY: node._AngeloPanY || 0,
            pointerId: event.pointerId,
            scaleX: (canvasWrap.clientWidth > 0 && wrapRect.width > 0)
                ? wrapRect.width / canvasWrap.clientWidth : 1,
            scaleY: (canvasWrap.clientHeight > 0 && wrapRect.height > 0)
                ? wrapRect.height / canvasWrap.clientHeight : 1,
        };
        if (node._AngeloCanvas) node._AngeloCanvas.style.cursor = "grabbing";
    });

    canvasWrap.addEventListener("pointermove", (event) => {
        const p = node._AngeloPanning;
        if (!p) return;
        node._AngeloPanX = p.startPanX + (event.clientX - p.startX) / (p.scaleX || 1);
        node._AngeloPanY = p.startPanY + (event.clientY - p.startY) / (p.scaleY || 1);
        applyView(node);
    });

    function endAngeloPan() {
        const p = node._AngeloPanning;
        if (!p) return;
        node._AngeloPanning = null;
        try { canvasWrap.releasePointerCapture(p.pointerId); } catch (e) { /* noop */ }
        redrawCanvasWithOverlays(node);   // restores the mode-appropriate cursor
    }
    canvasWrap.addEventListener("pointerup", (event) => {
        if (event.button === 1) endAngeloPan();
    });
    canvasWrap.addEventListener("pointercancel", endAngeloPan);

    // --- Right-click menu + drag-drop image loading (#7) ---
    // Right-click the preview → Open / Copy / Paste. The preview is a DOM
    // <canvas>, so we show our own LiteGraph menu here (the node's
    // getExtraMenuOptions only fires on the litegraph-rendered node body).
    canvasWrap.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        _angeloShowImageContextMenu(node, event);
    });

    // Drag-drop an OS image file onto the node → load it as the base, through
    // the SAME resolution popup + upload path as the Load Image button
    // (showLoadImagePopup). stopPropagation so the drop doesn't bubble to the
    // graph canvas (which would otherwise spawn a LoadImage node).
    const _dropHi = () => { container.style.outline = "2px dashed rgba(120,170,220,0.9)"; container.style.outlineOffset = "-2px"; };
    const _dropLo = () => { container.style.outline = ""; container.style.outlineOffset = ""; };
    container.addEventListener("dragover", (event) => {
        const dt = event.dataTransfer;
        if (dt && Array.from(dt.items || []).some((i) => i.kind === "file")) {
            event.preventDefault();
            event.stopPropagation();
            dt.dropEffect = "copy";
            _dropHi();
        }
    });
    container.addEventListener("dragleave", () => {
        // Clear unconditionally; a still-active drag re-adds it via dragover.
        _dropLo();
    });
    container.addEventListener("drop", (event) => {
        const dt = event.dataTransfer;
        const file = dt && dt.files && Array.from(dt.files).find((f) => f.type && f.type.startsWith("image/"));
        _dropLo();
        if (!file) return;
        event.preventDefault();
        event.stopPropagation();
        showLoadImagePopup(node, file);
    });

    // --- Pointer events (instead of mouse*) + pointer capture so long
    //     drags don't get cancelled when the cursor leaves the canvas
    //     boundary briefly. Pointer capture routes all subsequent move/
    //     up events to the canvas until the drag ends. ---

    // Helper: clamp event coordinates to the canvas's CSS rect, then
    // convert to canvas-intrinsic (= image-pixel) coordinates.
    function eventToImagePixel(event) {
        const rect = canvas.getBoundingClientRect();
        let cx = event.clientX - rect.left;
        let cy = event.clientY - rect.top;
        // Clamp so off-canvas drag positions still produce valid points
        // at the canvas edge rather than negative / oversized values.
        cx = Math.max(0, Math.min(rect.width, cx));
        cy = Math.max(0, Math.min(rect.height, cy));
        const img = node._AngeloImg;
        if (!img || !img.naturalWidth || rect.width === 0 || rect.height === 0) {
            return null;
        }
        return {
            cssX: cx,
            cssY: cy,
            pixelX: (cx / rect.width) * img.naturalWidth,
            pixelY: (cy / rect.height) * img.naturalHeight,
        };
    }

    canvas.addEventListener("pointermove", (event) => {
        const p = eventToImagePixel(event);
        if (!p) return;
        node._AngeloHover = { x: p.cssX, y: p.cssY };

        // Outpaint mode: edges are the direction picker, the interior is
        // the protect brush — an active drag stamps protect circles.
        if (isOutpaintMode(node)) {
            if (node._AngeloOutpaintPainting) {
                const prot = node._AngeloProtect = node._AngeloProtect || [];
                const last = prot[prot.length - 1];
                const r = _brushRadius(node);
                if (!last || Math.hypot(last[0] - p.pixelX, last[1] - p.pixelY) > r * 0.3) {
                    prot.push([p.pixelX, p.pixelY, r]);
                }
                redrawCanvasWithOverlays(node);
                return;
            }
            // Shift forces the protect brush everywhere — suppresses the
            // edge zone so strokes can START on something flush against
            // the frame edge (same convention as Detect's Shift brush).
            const dir = event.shiftKey ? null : _outpaintEdgeDir(node, p);
            if (dir !== node._AngeloOutpaintHoverDir) {
                node._AngeloOutpaintHoverDir = dir;
            }
            redrawCanvasWithOverlays(node);
            return;
        }

        // Detection select mode.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            // Active touch-up stroke → extend the brush along the drag.
            if (node._AngeloTouchup) {
                const tu = node._AngeloTouchup;
                _brushLine(tu.det, tu.last, [p.pixelX, p.pixelY], _brushRadius(node), tu.subtract);
                tu.last = [p.pixelX, p.pixelY];
                redrawCanvasWithOverlays(node);
                return;
            }
            // Brush preview while Shift/Alt is held (Refine only).
            const brushKey = (event.shiftKey || event.altKey)
                && !isSmartInpaintMode(node) && !isSmartGuidedInpaintMode(node);
            if (brushKey) {
                node._AngeloBrushPreview = { x: p.pixelX, y: p.pixelY, r: _brushRadius(node), subtract: event.altKey };
                canvas.style.cursor = "crosshair";
                node._AngeloHoverDet = -1;
                redrawCanvasWithOverlays(node);
                return;
            }
            if (node._AngeloBrushPreview) node._AngeloBrushPreview = null;
            // Otherwise highlight the candidate under the cursor.
            const det = _detAtPoint(node, p.pixelX, p.pixelY);
            const idx = det ? node._AngeloDetections.indexOf(det) : -1;
            canvas.style.cursor = idx >= 0 ? "pointer" : "default";
            if (idx !== node._AngeloHoverDet) {
                node._AngeloHoverDet = idx;
                redrawCanvasWithOverlays(node);
            }
            return;
        }

        if (node._AngeloDraggingRect) {
            // Smart Inpaint: update the live opposite-corner of the
            // drag-out rectangle as the user moves the cursor.
            node._AngeloDraggingRect.x2 = p.pixelX;
            node._AngeloDraggingRect.y2 = p.pixelY;
            node._AngeloDraggingRect.cssX2 = p.cssX;
            node._AngeloDraggingRect.cssY2 = p.cssY;
        } else if (node._AngeloPainting) {
            const stroke = node._AngeloStroke;
            const last = stroke[stroke.length - 1];
            // Dedup at 2px (image-pixel space) — saves bandwidth.
            if (!last || Math.hypot(last[0] - p.pixelX, last[1] - p.pixelY) > 2) {
                stroke.push([p.pixelX, p.pixelY]);
            }
        }
        redrawCanvasWithOverlays(node);
    });

    canvas.addEventListener("pointerleave", () => {
        node._AngeloHover = null;
        node._AngeloOutpaintHoverDir = null;
        if (node._AngeloBrushPreview) node._AngeloBrushPreview = null;
        // IMPORTANT: do NOT cancel an active paint stroke here. With
        // pointer capture set on pointerdown, we keep receiving move/up
        // events even when the cursor leaves the canvas — long strokes
        // can briefly cross the boundary without breaking.
        redrawCanvasWithOverlays(node);
        if (_AngeloHoveredNode === node) _AngeloHoveredNode = null;
    });

    // Track keyboard-shortcut hover ownership separately from the
    // paint-stroke pointer events above. pointerenter fires when the
    // cursor crosses INTO the canvas — at that point this node owns the
    // keyboard shortcuts.
    canvas.addEventListener("pointerenter", () => {
        _AngeloHoveredNode = node;
    });

    canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        // Detection select mode owns the canvas — let the click confirm a
        // candidate; don't start a paint stroke or rectangle drag. EXCEPT a
        // Shift/Alt drag, which is the touch-up brush (Refine only): Shift
        // adds to the first-overlapped candidate's mask, Alt subtracts.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            const brushKey = (event.shiftKey || event.altKey)
                && !isSmartInpaintMode(node) && !isSmartGuidedInpaintMode(node);
            if (brushKey) {
                const pp = eventToImagePixel(event);
                if (!pp) return;
                const target = _pickTouchupTarget(node, pp.pixelX, pp.pixelY);
                if (!target) return;
                try { canvas.setPointerCapture(event.pointerId); node._AngeloPointerId = event.pointerId; } catch (e) { /* noop */ }
                const subtract = event.altKey;
                _ensureEditMask(node, target);
                _brushStamp(target, pp.pixelX, pp.pixelY, _brushRadius(node), subtract);
                node._AngeloTouchup = { det: target, subtract, last: [pp.pixelX, pp.pixelY] };
                node._AngeloBrushPreview = null;
                redrawCanvasWithOverlays(node);
            }
            return;
        }
        // Outpaint mode: near an edge the click handler owns the extension;
        // in the interior, a drag paints the PROTECT brush (areas the
        // overlap band must leave frozen).
        if (isOutpaintMode(node)) {
            // Shift = protect brush wins even in the edge zone.
            if (node._AngeloOutpaintHoverDir && !event.shiftKey) return;  // edge-click = extend
            node._AngeloOutpaintHoverDir = null;
            const pp = eventToImagePixel(event);
            if (!pp) return;
            try {
                canvas.setPointerCapture(event.pointerId);
                node._AngeloPointerId = event.pointerId;
            } catch (e) { /* noop */ }
            node._AngeloOutpaintPainting = true;
            node._AngeloProtect = node._AngeloProtect || [];
            node._AngeloProtect.push([pp.pixelX, pp.pixelY, _brushRadius(node)]);
            redrawCanvasWithOverlays(node);
            return;
        }
        // Smart Guided Inpaint has no canvas interaction at all — the
        // location comes from the dropdown, the run from the button.
        if (isSmartGuidedInpaintMode(node)) return;
        const smartInpaint = isSmartInpaintMode(node);
        const paintOn = isPaintModeOn(node);
        dbg("pointerdown", { smartInpaint, paintModeOn: paintOn });

        // Smart Inpaint always owns the canvas — drag-out rectangle.
        // It takes priority over paint_mode.
        if (!smartInpaint && !paintOn) return;
        const p = eventToImagePixel(event);
        if (!p) return;

        // Pointer capture: route all subsequent pointer events to this
        // element until pointerup, regardless of cursor position.
        try {
            canvas.setPointerCapture(event.pointerId);
            node._AngeloPointerId = event.pointerId;
        } catch (e) {
            dbg("setPointerCapture failed", e);
        }

        if (smartInpaint) {
            node._AngeloDraggingRect = {
                x1: p.pixelX, y1: p.pixelY,
                x2: p.pixelX, y2: p.pixelY,
                cssX1: p.cssX, cssY1: p.cssY,
                cssX2: p.cssX, cssY2: p.cssY,
            };
        } else {
            node._AngeloPainting = true;
            node._AngeloStroke = [[p.pixelX, p.pixelY]];
        }
        redrawCanvasWithOverlays(node);
        // Don't preventDefault on pointerdown — we still want pointermove
        // to fire normally. The subsequent "click" event won't fire if
        // any meaningful drag occurred (browser behaviour), and our
        // click handler also short-circuits when paint_mode is on.
    });

    function endPaintStroke(event) {
        if (!node._AngeloPainting) return;
        node._AngeloPainting = false;
        if (node._AngeloPointerId !== undefined) {
            try { canvas.releasePointerCapture(node._AngeloPointerId); }
            catch (e) { /* already released */ }
            node._AngeloPointerId = undefined;
        }
        const stroke = node._AngeloStroke || [];
        node._AngeloStroke = null;
        if (stroke.length === 0) {
            redrawCanvasWithOverlays(node);
            return;
        }
        dbg("paint stroke submitted", { points: stroke.length });
        triggerPaintRefine(node, stroke);
        redrawCanvasWithOverlays(node);
    }

    function endRectDrag() {
        if (!node._AngeloDraggingRect) return;
        const r = node._AngeloDraggingRect;
        node._AngeloDraggingRect = null;
        if (node._AngeloPointerId !== undefined) {
            try { canvas.releasePointerCapture(node._AngeloPointerId); }
            catch (e) { /* already released */ }
            node._AngeloPointerId = undefined;
        }
        const dx = Math.abs(r.x2 - r.x1);
        const dy = Math.abs(r.y2 - r.y1);
        // Reject degenerate (single-click) drags — Smart Inpaint needs
        // an actual rectangle. Threshold in image-pixel space.
        if (dx < 8 || dy < 8) {
            dbg("rect drag too small — ignored", { dx, dy });
            redrawCanvasWithOverlays(node);
            return;
        }
        dbg("smart inpaint rect submitted", r);
        triggerRectRefine(node, [r.x1, r.y1, r.x2, r.y2]);
        redrawCanvasWithOverlays(node);
    }

    function endTouchup() {
        if (!node._AngeloTouchup) return;
        node._AngeloTouchup = null;
        if (node._AngeloPointerId !== undefined) {
            try { canvas.releasePointerCapture(node._AngeloPointerId); }
            catch (e) { /* already released */ }
            node._AngeloPointerId = undefined;
        }
        redrawCanvasWithOverlays(node);
    }

    function endOutpaintProtect() {
        if (!node._AngeloOutpaintPainting) return;
        node._AngeloOutpaintPainting = false;
        if (node._AngeloPointerId !== undefined) {
            try { canvas.releasePointerCapture(node._AngeloPointerId); }
            catch (e) { /* already released */ }
            node._AngeloPointerId = undefined;
        }
        syncOutpaintControls(node);   // refresh the Clear-protect chip count
        redrawCanvasWithOverlays(node);
    }

    canvas.addEventListener("pointerup", (event) => {
        if (event.button !== 0) return;
        if (node._AngeloOutpaintPainting) endOutpaintProtect();
        else if (node._AngeloTouchup) endTouchup();
        else if (node._AngeloDraggingRect) endRectDrag();
        else endPaintStroke(event);
    });
    canvas.addEventListener("pointercancel", (event) => {
        if (node._AngeloOutpaintPainting) endOutpaintProtect();
        else if (node._AngeloTouchup) endTouchup();
        else if (node._AngeloDraggingRect) endRectDrag();
        else endPaintStroke(event);
    });

    // --- Single-click refine (click mode only — paint mode and
    //     Smart Inpaint handle the canvas via pointer drag above). ---
    canvas.addEventListener("click", (event) => {
        // Detection select mode owns the click (Refine + Smart Inpaint):
        // clicking a candidate edits it and keeps the rest up for more
        // edits. Empty-space clicks do nothing (so you can't accidentally
        // exit mid-batch) — leave via Cancel Detect / Esc / Space.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            // Shift/Alt click is the touch-up brush, not a confirm.
            if (event.shiftKey || event.altKey) return;
            const p = eventToImagePixel(event);
            const det = p ? _detAtPoint(node, p.pixelX, p.pixelY) : null;
            if (det) confirmDetection(node, det);
            return;
        }
        // Outpaint mode: a click near an edge extends the canvas that way.
        // Shift-clicks belong to the protect brush, never the extension.
        if (isOutpaintMode(node)) {
            if (node._AngeloOutpaintHoverDir && !event.shiftKey) {
                triggerOutpaint(node, node._AngeloOutpaintHoverDir);
            }
            return;
        }
        if (isSmartGuidedInpaintMode(node)) return; // no canvas interaction
        if (isSmartInpaintMode(node)) return; // rectangle-drag owns it
        if (isPaintModeOn(node)) return; // paint mode owns the interaction
        const rect = canvas.getBoundingClientRect();
        const cx = event.clientX - rect.left;
        const cy = event.clientY - rect.top;
        const img = node._AngeloImg;
        if (!img || !img.naturalWidth) {
            dbg("click ignored — no image loaded yet");
            return;
        }
        const pixelX = Math.floor((cx / rect.width) * img.naturalWidth);
        const pixelY = Math.floor((cy / rect.height) * img.naturalHeight);
        dbg("click", { cx, cy, pixelX, pixelY });
        triggerRefine(node, pixelX, pixelY, cx, cy);
        flashClickOverlay(node, cx, cy);
    });

    // Add as a DOM widget on the node so LiteGraph manages its layout.
    // getMinHeight floors the DOM-widget area; the toolbar now spans 4
    // rows so we give the canvas a sensible minimum below it. The canvas
    // itself scales to fill whatever space is left (fitCanvasDisplaySize),
    // so resizing the node taller just grows the image.
    const widget = node.addDOMWidget("Angelo_preview_canvas", "Angelo_canvas", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 320,
    });

    // Make the preview widget fill the node's full width.
    //
    // LiteGraph computes a DOM widget's wrapper-element width as
    //     (widget.width ?? node.width) - 2 * margin
    // (see the Vue node renderer). The `?? node.width` fallback gives full
    // width ONLY while `widget.width` is unset; once LiteGraph's widget-draw
    // path assigns `widget.width` (it sets it to the narrow content-min when
    // the node's hidden widgets report ~340), that pinned value wins. So a
    // node dragged wide shows a wide frame but the preview collapses to ~340
    // on the left, and any relayout (e.g. clicking a control) re-pins it.
    // Defining `width` as a getter that always returns the live node width
    // keeps the wrapper at full width and ignores the narrow re-pin. Setter is
    // a no-op so LiteGraph's assignment can't shrink it again.
    try {
        Object.defineProperty(widget, "width", {
            configurable: true,
            enumerable: true,
            get() { return node.size ? node.size[0] : undefined; },
            set(_v) { /* ignore — width is derived from the node, see above */ },
        });
    } catch (e) {
        dbg("could not pin widget.width getter", e);
    }

    node._AngeloWidget = widget;
    node._AngeloCanvas = canvas;
    node._AngeloCanvasWrap = canvasWrap;
    node._AngeloPlaceholder = placeholder;
    node._AngeloContainer = container;

    // Re-fit the canvas whenever its available area changes — node
    // resize, area-prompt box show/hide, etc. ResizeObserver fires on
    // the wrap's rendered-size changes, which covers all of them.
    if (typeof ResizeObserver !== "undefined") {
        const ro = new ResizeObserver(() => fitCanvasDisplaySize(node));
        ro.observe(canvasWrap);
        node._AngeloCanvasResizeObserver = ro;
        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            try { ro.disconnect(); } catch (e) { /* noop */ }
            // Deleting the node mid-fullscreen would otherwise leave the
            // overlay (and its detached container) orphaned in the body.
            try { if (node._AngeloFSOverlay) exitAngeloFullscreen(node); } catch (e) { /* noop */ }
            if (onRemoved) onRemoved.apply(this, arguments);
        };
    }
    fitCanvasDisplaySize(node);

    dbg("attached preview canvas to node", node.id);
}

function _angeloIsZoomed(node) {
    // Epsilon so float drift near 1.0 doesn't read as "zoomed".
    return Math.abs((node._AngeloZoom || 1) - 1) > 1e-3
        || (node._AngeloPanX || 0) !== 0
        || (node._AngeloPanY || 0) !== 0;
}

// Compute the BASE (fit) display size: the canvas size at zoom=1, fitting
// the wrap while preserving aspect ratio. Stored as _AngeloBaseW/H; the
// live size is base × zoom, applied by applyView().
//
// IMPORTANT: when the user has zoomed/panned (zoom != 1 or pan != 0), this
// is a NO-OP — the auto-fit must not stomp on a manual zoom. The view only
// re-fits at the neutral state (e.g. after resetView or a new image).
function fitCanvasDisplaySize(node) {
    if (_angeloIsZoomed(node)) return;   // never auto-fit while zoomed/panned
    const canvas = node._AngeloCanvas;
    const wrap = node._AngeloCanvasWrap;
    if (!canvas || !wrap) return;
    const availW = wrap.clientWidth;
    const availH = wrap.clientHeight;
    if (availW <= 0 || availH <= 0) return;
    const img = node._AngeloImg;
    const natW = (img && img.naturalWidth) ? img.naturalWidth : canvas.width;
    const natH = (img && img.naturalHeight) ? img.naturalHeight : canvas.height;
    if (natW <= 0 || natH <= 0) return;
    const scale = Math.min(availW / natW, availH / natH);
    node._AngeloBaseW = Math.max(1, Math.floor(natW * scale));
    node._AngeloBaseH = Math.max(1, Math.floor(natH * scale));
    applyView(node);
}

// Apply the current zoom/pan: size + position the absolutely-placed
// canvas. Display size = base × zoom; positioned centred in the wrap with
// the pan offset added. Then refresh the minimap.
function applyView(node) {
    const canvas = node._AngeloCanvas;
    const wrap = node._AngeloCanvasWrap;
    if (!canvas || !wrap) return;
    const baseW = node._AngeloBaseW, baseH = node._AngeloBaseH;
    if (!baseW || !baseH) return;
    const z = node._AngeloZoom || 1;
    const dispW = baseW * z, dispH = baseH * z;
    const wrapW = wrap.clientWidth, wrapH = wrap.clientHeight;
    const left = (wrapW - dispW) / 2 + (node._AngeloPanX || 0);
    const top = (wrapH - dispH) / 2 + (node._AngeloPanY || 0);
    canvas.style.width = dispW + "px";
    canvas.style.height = dispH + "px";
    canvas.style.left = left + "px";
    canvas.style.top = top + "px";
    updateMinimap(node);
}

// Reset to the neutral fit view (zoom=1, no pan) and re-fit to the node.
function resetView(node) {
    node._AngeloZoom = 1;
    node._AngeloPanX = 0;
    node._AngeloPanY = 0;
    fitCanvasDisplaySize(node);  // recomputes base + applyView (now neutral)
}

// Draw the corner minimap: full-image thumbnail + a rectangle marking the
// currently-visible viewport. Shown only when zoomed in (zoom > 1).
function updateMinimap(node) {
    const mm = node._AngeloMinimap;
    const wrap = node._AngeloCanvasWrap;
    const img = node._AngeloImg;
    if (!mm || !wrap) return;
    const z = node._AngeloZoom || 1;
    // Epsilon so a near-1.0 (drifted) zoom doesn't flash the minimap.
    if (z <= 1.001 || !img || !img.naturalWidth) {
        mm.style.display = "none";
        return;
    }
    // Thumbnail size: cap the long edge at 140 px, preserve aspect.
    const natW = img.naturalWidth, natH = img.naturalHeight;
    const cap = 140;
    const mmScale = Math.min(cap / natW, cap / natH);
    const mmW = Math.max(1, Math.round(natW * mmScale));
    const mmH = Math.max(1, Math.round(natH * mmScale));
    if (mm.width !== mmW) mm.width = mmW;
    if (mm.height !== mmH) mm.height = mmH;
    mm.style.width = mmW + "px";
    mm.style.height = mmH + "px";
    mm.style.display = "block";

    const ctx = mm.getContext("2d");
    ctx.clearRect(0, 0, mmW, mmH);
    ctx.drawImage(img, 0, 0, mmW, mmH);

    // Visible image region (normalised 0..1) = which part of the canvas
    // currently lands inside the wrap viewport.
    const baseW = node._AngeloBaseW, baseH = node._AngeloBaseH;
    const dispW = baseW * z, dispH = baseH * z;
    const wrapW = wrap.clientWidth, wrapH = wrap.clientHeight;
    const left = (wrapW - dispW) / 2 + (node._AngeloPanX || 0);
    const top = (wrapH - dispH) / 2 + (node._AngeloPanY || 0);
    const vx0 = Math.max(0, Math.min(1, (0 - left) / dispW));
    const vy0 = Math.max(0, Math.min(1, (0 - top) / dispH));
    const vx1 = Math.max(0, Math.min(1, (wrapW - left) / dispW));
    const vy1 = Math.max(0, Math.min(1, (wrapH - top) / dispH));

    ctx.strokeStyle = "rgba(255, 220, 80, 0.95)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(vx0 * mmW, vy0 * mmH, (vx1 - vx0) * mmW, (vy1 - vy0) * mmH);
    ctx.fillStyle = "rgba(255, 220, 80, 0.12)";
    ctx.fillRect(vx0 * mmW, vy0 * mmH, (vx1 - vx0) * mmW, (vy1 - vy0) * mmH);
}

// Area Prompt box: a single textarea below the canvas plus a Pos/Neg
// toggle. The toggle decides which underlying widget the textarea is
// bound to (area_text_positive / area_text_negative). We keep the
// non-edited prompt's value in its widget untouched, so flipping the
// toggle just re-points the textarea at the other widget's text.
function attachAreaPromptBox(node, container) {
    node._AngeloAreaPromptTarget = "positive";

    const wrap = document.createElement("div");
    wrap.style.display = "flex";  // toggled to "none" by syncAreaPromptVisibility
    wrap.style.flexDirection = "column";
    wrap.style.background = "#222";
    wrap.style.borderTop = "1px solid #333";
    wrap.style.borderBottom = "1px solid #333";
    wrap.style.padding = "4px";
    wrap.style.gap = "4px";

    // Header row: label + Pos/Neg toggle.
    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "6px";

    const label = document.createElement("span");
    label.textContent = "Area Prompt";
    label.style.color = "#bbb";
    label.style.fontSize = "11px";
    label.style.userSelect = "none";
    header.appendChild(label);

    // Prompt slots (#12): six numbered presets, each holding its own
    // positive + negative Area Prompt. Click a number to switch — the
    // current text is stashed into the outgoing slot first, so nothing
    // is ever lost. Persisted (with the active index) in the hidden
    // area_prompt_slots widget, so slots survive workflow save/load.
    const slotStrip = document.createElement("div");
    slotStrip.style.cssText = "display:flex; gap:3px; align-items:center;";
    slotStrip.title = "Prompt slots — six independent Area Prompts saved with the workflow. "
        + "Click a number to switch (the current text is kept in its slot). "
        + "E.g. slot 1 = 'mushrooms', slot 2 = 'bones': paint, switch, paint.";
    const slotBtns = [];
    for (let i = 0; i < _ANGELO_NUM_SLOTS; i++) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = String(i + 1);
        b.style.cssText = "width:21px; height:19px; padding:0; font-size:10px; font-weight:bold; "
            + "border:1px solid #555; border-radius:3px; background:#2a2a2a; color:#888; "
            + "cursor:pointer; line-height:1; user-select:none;";
        for (const ev of ["pointerdown", "mousedown"]) {
            b.addEventListener(ev, (e) => e.stopPropagation());
        }
        b.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectPromptSlot(node, i);
        });
        slotStrip.appendChild(b);
        slotBtns.push(b);
    }
    header.appendChild(slotStrip);
    node._AngeloPromptSlotBtns = slotBtns;

    const posNegBtn = document.createElement("button");
    posNegBtn.textContent = "Positive";
    posNegBtn.style.fontSize = "11px";
    posNegBtn.style.padding = "2px 8px";
    posNegBtn.style.borderRadius = "3px";
    posNegBtn.style.border = "1px solid #555";
    posNegBtn.style.background = "rgba(30, 120, 80, 0.95)";
    posNegBtn.style.color = "#fff";
    posNegBtn.style.cursor = "pointer";
    posNegBtn.style.userSelect = "none";
    posNegBtn.title = "Switch which Area Prompt this box edits. Positive = what to draw; "
        + "Negative = what to avoid (ignored by CFG=1 / distilled models like Klein, kept for others).";
    header.appendChild(posNegBtn);

    const textarea = document.createElement("textarea");
    textarea.placeholder = "Area prompt — describes the masked region (used when Area Prompt is ON).";
    textarea.style.width = "100%";
    textarea.style.boxSizing = "border-box";
    textarea.style.minHeight = "48px";
    textarea.style.resize = "vertical";
    textarea.style.fontSize = "12px";
    textarea.style.fontFamily = "inherit";
    textarea.style.padding = "4px";
    textarea.style.border = "1px solid #555";
    textarea.style.borderRadius = "3px";
    textarea.style.background = "#1a1a1a";
    textarea.style.color = "#ddd";

    // Stop pointer/key events from bubbling to the graph canvas (node
    // drag / delete / canvas shortcuts) while editing.
    for (const ev of ["pointerdown", "mousedown", "keydown", "keyup", "wheel"]) {
        textarea.addEventListener(ev, (e) => e.stopPropagation());
    }

    const targetWidgetName = () =>
        node._AngeloAreaPromptTarget === "negative" ? "area_text_negative" : "area_text_positive";

    textarea.addEventListener("input", () => {
        const w = findWidget(node, targetWidgetName());
        if (w) setWidget(w, textarea.value);
        // Mirror into the active prompt slot so slots survive switches
        // and workflow saves without a separate "save slot" action.
        _angeloPersistActiveSlot(node);
        // Keep the Outpaint combined-prompt preview live while typing.
        syncOutpaintPromptPreview(node);
    });

    posNegBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        // Persist current text before switching (input handler already
        // did, but be safe), then flip the target and reload.
        const curW = findWidget(node, targetWidgetName());
        if (curW) setWidget(curW, textarea.value);

        node._AngeloAreaPromptTarget =
            node._AngeloAreaPromptTarget === "negative" ? "positive" : "negative";
        syncAreaPromptBox(node);
    });

    // "Insert Smart Phrasing" — opens a popup of edit-preservation
    // constraints; ticked ones get appended to the active Area Prompt.
    const smartBtn = document.createElement("button");
    smartBtn.type = "button";
    smartBtn.textContent = "Insert Smart Phrasing";
    smartBtn.style.cssText = "align-self:flex-start; font-size:11px; padding:3px 10px; "
        + "border:1px solid #555; border-radius:3px; background:#2a2a2a; color:#bbb; cursor:pointer;";
    smartBtn.title = "Append edit-preservation phrases (keep lighting / pose / clothes / faces the same) to the Area Prompt above.";
    for (const ev of ["pointerdown", "mousedown"]) {
        smartBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    smartBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        showSmartPhrasingPopup(node);
    });

    // Location dropdown (Smart Guided Inpaint only) — sits below the
    // "Area Prompt" heading and above the textarea. Picks the spatial
    // prefix prepended to the prompt at run time.
    const locationSelect = makeDropdown("Location", _Angelo_GUIDED_LOCATIONS, (val) => {
        const w = findWidget(node, "guided_location");
        if (w) setWidget(w, val);
    });
    locationSelect.title = "Where to place the new content. Prepended to your prompt at run time "
        + "(e.g. 'In the top left of the image, ...'). Smart Guided Inpaint only.";
    locationSelect.style.padding = "0";

    // "Generate Guided Edit" run button (Smart Guided Inpaint only) —
    // there's no click/drag to trigger a run in this mode.
    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.textContent = "Generate Guided Edit";
    runBtn.style.cssText = "align-self:center; font-size:11px; padding:4px 12px; "
        + "border:1px solid #4a7; border-radius:3px; background:rgba(30,120,80,0.95); "
        + "color:#fff; font-weight:bold; cursor:pointer;";
    runBtn.title = "Run the whole-image guided edit using the Location + Area Prompt.";
    for (const ev of ["pointerdown", "mousedown"]) {
        runBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    runBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        triggerGuidedRefine(node);
    });

    // Outpaint prompt review — shown only in Outpaint mode. An order
    // selector (instruction first vs my text first; some models weight
    // the head of the prompt more) plus a live read-only preview of the
    // EXACT combined prompt that will be encoded, so there's no guessing
    // what the instruction did to your text.
    const opPromptWrap = document.createElement("div");
    opPromptWrap.style.cssText = "display:none; flex-direction:column; gap:3px; "
        + "padding:2px 0 0 0;";
    const opOrderRow = document.createElement("div");
    opOrderRow.style.cssText = "display:flex; align-items:center; gap:6px;";
    const opOrderSelect = makeDropdown("Order", ["Instruction first", "My text first"], (val) => {
        const w = findWidget(node, "outpaint_instruction_pos");
        if (w) setWidget(w, val === "My text first" ? "append" : "prepend");
        syncOutpaintPromptPreview(node);
    });
    opOrderSelect.title = "Where Angelo's extend-the-scene instruction sits relative to your text. "
        + "Instruction first is the default; try 'My text first' if the model is leaning too hard "
        + "on the instruction and under-weighting your description (some models weight the start "
        + "of the prompt more heavily). The preview below shows exactly what gets encoded.";
    opOrderRow.appendChild(opOrderSelect);
    const opPreviewLabel = document.createElement("span");
    opPreviewLabel.textContent = "Final prompt:";
    opPreviewLabel.style.cssText = "font-size:10px; color:#8aa;";
    opOrderRow.appendChild(opPreviewLabel);
    const opPromptPreview = document.createElement("div");
    opPromptPreview.style.cssText = "font-size:10px; font-style:italic; color:#9ab; "
        + "background:#1a1a1a; border:1px solid #3a3a3a; border-radius:3px; "
        + "padding:4px 6px; white-space:pre-wrap; word-break:break-word; "
        + "max-height:64px; overflow-y:auto; user-select:text;";
    opPromptPreview.title = "The exact prompt the outpaint will encode (direction phrase follows "
        + "the last-used / next-clicked direction).";
    for (const ev of ["pointerdown", "mousedown", "wheel"]) {
        opPromptPreview.addEventListener(ev, (e) => e.stopPropagation());
    }
    opPromptWrap.appendChild(opOrderRow);
    opPromptWrap.appendChild(opPromptPreview);
    node._AngeloOutpaintPromptWrap = opPromptWrap;
    node._AngeloOutpaintOrderSelect = opOrderSelect;
    node._AngeloOutpaintPromptPreview = opPromptPreview;

    wrap.appendChild(header);
    wrap.appendChild(locationSelect);
    wrap.appendChild(textarea);
    wrap.appendChild(opPromptWrap);
    wrap.appendChild(smartBtn);
    wrap.appendChild(runBtn);
    container.appendChild(wrap);

    node._AngeloAreaPromptWrap = wrap;
    node._AngeloAreaPromptTextarea = textarea;
    node._AngeloAreaPromptPosNegBtn = posNegBtn;
    node._AngeloAreaPromptLabel = label;
    node._AngeloSmartPhrasingBtn = smartBtn;
    node._AngeloGuidedLocationSelect = locationSelect;
    node._AngeloGuidedRunBtn = runBtn;

    syncAreaPromptBox(node);
    syncAreaPromptVisibility(node);
}

// Edit-preservation phrases for the Smart Phrasing popup. Adding more
// here automatically adds a checkbox.
const _Angelo_SMART_PHRASES = [
    "Keep the lighting the same",
    "Keep the pose the same",
    "Keep the clothes the same",
    "Keep the faces the same",
];

// Append the chosen phrases to the currently-active Area Prompt textarea
// (positive or negative per the Pos/Neg toggle), comma-joined onto any
// existing text. Skips phrases already present (case-insensitive) so
// re-opening the popup doesn't duplicate them. Persists to the widget.
function appendSmartPhrases(node, phrases) {
    const textarea = node._AngeloAreaPromptTextarea;
    if (!textarea || !phrases.length) return;
    const cur = textarea.value.trim();
    const lower = cur.toLowerCase();
    const fresh = phrases.filter((p) => !lower.includes(p.toLowerCase()));
    if (!fresh.length) return;
    const addition = fresh.join(", ");
    const next = cur ? (cur.replace(/,\s*$/, "") + ", " + addition) : addition;
    textarea.value = next;
    const wname = node._AngeloAreaPromptTarget === "negative"
        ? "area_text_negative" : "area_text_positive";
    const w = findWidget(node, wname);
    if (w) setWidget(w, next);
    _angeloPersistActiveSlot(node);
}

// ===== Prompt slots (#12) — six numbered Area Prompt presets ============
// State shape (JSON in the hidden area_prompt_slots widget):
//   { active: 0-5, slots: [{pos, neg} × 6] }
// area_text_positive/negative stay the encode source of truth; the slots
// are a UI-level store the active slot mirrors into / out of.

const _ANGELO_NUM_SLOTS = 6;

function _angeloDefaultSlots() {
    return {
        active: 0,
        slots: Array.from({ length: _ANGELO_NUM_SLOTS }, () => ({ pos: "", neg: "" })),
    };
}

function _angeloLoadSlots(node) {
    const w = findWidget(node, "area_prompt_slots");
    let data = null;
    try { data = JSON.parse(String(w?.value || "")); } catch (e) { /* fresh */ }
    if (!data || !Array.isArray(data.slots) || data.slots.length !== _ANGELO_NUM_SLOTS) {
        data = _angeloDefaultSlots();
        // Seed slot 1 from whatever the area_text widgets already hold so
        // upgrading users keep their existing prompt.
        data.slots[0].pos = String(findWidget(node, "area_text_positive")?.value || "");
        data.slots[0].neg = String(findWidget(node, "area_text_negative")?.value || "");
    }
    data.active = Math.max(0, Math.min(_ANGELO_NUM_SLOTS - 1, Number(data.active) || 0));
    for (let i = 0; i < _ANGELO_NUM_SLOTS; i++) {
        if (!data.slots[i] || typeof data.slots[i] !== "object") data.slots[i] = { pos: "", neg: "" };
        data.slots[i].pos = String(data.slots[i].pos || "");
        data.slots[i].neg = String(data.slots[i].neg || "");
    }
    return data;
}

function _angeloSaveSlots(node, data) {
    const w = findWidget(node, "area_prompt_slots");
    if (w) setWidget(w, JSON.stringify(data));
}

// Stash the current area_text widget values into the active slot.
function _angeloPersistActiveSlot(node) {
    const data = _angeloLoadSlots(node);
    data.slots[data.active].pos = String(findWidget(node, "area_text_positive")?.value || "");
    data.slots[data.active].neg = String(findWidget(node, "area_text_negative")?.value || "");
    _angeloSaveSlots(node, data);
    syncPromptSlotButtons(node);
}

// Switch the active slot: persist the outgoing slot, load the incoming
// one into the area_text widgets + textarea, restyle the buttons.
function selectPromptSlot(node, idx) {
    const data = _angeloLoadSlots(node);
    if (idx === data.active) return;
    data.slots[data.active].pos = String(findWidget(node, "area_text_positive")?.value || "");
    data.slots[data.active].neg = String(findWidget(node, "area_text_negative")?.value || "");
    data.active = idx;
    const slot = data.slots[idx];
    const wp = findWidget(node, "area_text_positive");
    const wn = findWidget(node, "area_text_negative");
    if (wp) setWidget(wp, slot.pos);
    if (wn) setWidget(wn, slot.neg);
    _angeloSaveSlots(node, data);
    syncAreaPromptBox(node);
    syncPromptSlotButtons(node);
}

// Active slot = purple (matches the Area Prompt toggle); slots holding
// text show brighter digits than empty ones so you can see at a glance
// which presets are loaded.
function syncPromptSlotButtons(node) {
    const btns = node._AngeloPromptSlotBtns;
    if (!btns) return;
    const data = _angeloLoadSlots(node);
    btns.forEach((b, i) => {
        const active = i === data.active;
        const filled = !!(data.slots[i].pos || data.slots[i].neg);
        b.style.background = active ? "rgba(95, 50, 130, 0.95)" : "#2a2a2a";
        b.style.color = active ? "#fff" : (filled ? "#ccc" : "#777");
        b.style.borderColor = active ? "rgba(180, 140, 220, 0.9)" : "#555";
    });
}

// Body-level modal popups (Smart Phrasing / Load Image / Shrink) normally
// attach to document.body so they overlay the whole page. But while the node is
// in fullscreen, the fullscreen overlay sits ABOVE document.body's popups, so
// they'd be hidden until you exit. Parent them into the overlay instead when
// it's active — the overlay owns the stacking context, so a fixed backdrop
// inside it still covers the viewport and renders above the editor.
function _angeloModalParent(node) {
    return (node && node._AngeloFSOverlay) ? node._AngeloFSOverlay : document.body;
}

// Attach a modal backdrop: tags it (so the fullscreen Esc handler can defer
// to it, and exitAngeloFullscreen can rescue it), appends it to the right
// parent, and wires a generic capture-phase Esc that closes the popup. The
// listener self-removes when the backdrop is gone regardless of HOW it was
// closed (button, backdrop click, fullscreen teardown), so nothing leaks.
function _angeloShowModal(node, backdrop) {
    backdrop.classList.add("angelo-modal-backdrop");
    const onKey = (e) => {
        if (!backdrop.isConnected) {
            document.removeEventListener("keydown", onKey, true);
            return;
        }
        if (e.key !== "Escape") return;
        e.preventDefault();
        e.stopImmediatePropagation();
        document.removeEventListener("keydown", onKey, true);
        backdrop.remove();
    };
    document.addEventListener("keydown", onKey, true);
    _angeloModalParent(node).appendChild(backdrop);
}

function showSmartPhrasingPopup(node) {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed; inset:0; background:rgba(0,0,0,0.6); "
        + "display:flex; align-items:center; justify-content:center; z-index:10000; "
        + "font-family:Arial,sans-serif;";

    const modal = document.createElement("div");
    modal.style.cssText = "background:#2a2a2a; color:#ddd; border:1px solid #555; "
        + "border-radius:8px; padding:16px; width:340px; max-width:90vw; "
        + "display:flex; flex-direction:column; gap:8px;";

    const header = document.createElement("div");
    header.textContent = "Insert Smart Phrasing";
    header.style.cssText = "font-size:14px; font-weight:bold; color:#aaa;";
    modal.appendChild(header);

    const hint = document.createElement("div");
    hint.textContent = "Tick the constraints to add to the Area Prompt.";
    hint.style.cssText = "font-size:11px; color:#888; margin-bottom:4px;";
    modal.appendChild(hint);

    const checks = [];
    for (const phrase of _Angelo_SMART_PHRASES) {
        const row = document.createElement("label");
        row.style.cssText = "display:flex; align-items:center; gap:8px; font-size:13px; "
            + "cursor:pointer; padding:3px 2px;";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        const span = document.createElement("span");
        span.textContent = phrase;
        row.appendChild(cb);
        row.appendChild(span);
        modal.appendChild(row);
        checks.push({ cb, phrase });
    }

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex; justify-content:flex-end; gap:8px; margin-top:8px;";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = "background:#444; color:#ddd; border:none; padding:6px 14px; "
        + "border-radius:4px; cursor:pointer;";

    const insertBtn = document.createElement("button");
    insertBtn.textContent = "Insert";
    insertBtn.style.cssText = "background:rgba(30,120,80,0.95); color:#fff; border:none; "
        + "padding:6px 14px; border-radius:4px; cursor:pointer;";

    footer.appendChild(cancelBtn);
    footer.appendChild(insertBtn);
    modal.appendChild(footer);
    backdrop.appendChild(modal);
    _angeloShowModal(node, backdrop);

    const close = () => { if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop); };
    insertBtn.addEventListener("click", () => {
        const chosen = checks.filter((c) => c.cb.checked).map((c) => c.phrase);
        appendSmartPhrases(node, chosen);
        close();
    });
    cancelBtn.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
    // Esc-to-close is handled generically by _angeloShowModal (its listener
    // self-removes with the backdrop — the bespoke one here used to leak).
}

// Reflect the current edit-target widget value into the textarea and
// update the Pos/Neg button styling. Called on creation, on toggle,
// and from the full toolbar sync (workflow load).
function syncAreaPromptBox(node) {
    const textarea = node._AngeloAreaPromptTextarea;
    const btn = node._AngeloAreaPromptPosNegBtn;
    if (!textarea || !btn) return;
    const onNeg = node._AngeloAreaPromptTarget === "negative";
    const wname = onNeg ? "area_text_negative" : "area_text_positive";
    const w = findWidget(node, wname);
    const val = w ? String(w.value ?? "") : "";
    if (textarea.value !== val) textarea.value = val;
    btn.textContent = onNeg ? "Negative" : "Positive";
    btn.style.background = onNeg ? "rgba(140, 60, 60, 0.95)" : "rgba(30, 120, 80, 0.95)";
}

// Show the Area Prompt box only when Area Prompt is effectively ON —
// either the area_prompt widget is true, or Smart Inpaint mode forces
// it on. Hiding is display:none only; the text lives in the
// area_text_* widgets and is reloaded by syncAreaPromptBox, so toggling
// visibility never loses what was typed.
function syncAreaPromptVisibility(node) {
    const guided = isSmartGuidedInpaintMode(node);
    const anySmart = isSmartInpaintMode(node) || guided;

    // Smart Phrasing button: both smart modes (both use reference-image
    // conditioning, so the "keep X the same" constraints apply).
    if (node._AngeloSmartPhrasingBtn) {
        node._AngeloSmartPhrasingBtn.style.display = anySmart ? "block" : "none";
    }
    // Location dropdown + Generate button: Smart Guided Inpaint only.
    if (node._AngeloGuidedLocationSelect) {
        node._AngeloGuidedLocationSelect.style.display = guided ? "inline-flex" : "none";
    }
    if (node._AngeloGuidedRunBtn) {
        node._AngeloGuidedRunBtn.style.display = guided ? "block" : "none";
    }

    const wrap = node._AngeloAreaPromptWrap;
    if (!wrap) return;
    const w = findWidget(node, "area_prompt");
    const on = (w && !!w.value) || anySmart;
    const next = on ? "flex" : "none";
    if (wrap.style.display === next) return;  // no change → no reflow
    wrap.style.display = next;
    // The container's height just changed — nudge LiteGraph to recompute
    // the node's reserved space for the DOM widget so the box isn't clipped.
    if (node.graph && node.graph.setDirtyCanvas) node.graph.setDirtyCanvas(true, true);
}

function loadIntoCanvas(node, url) {
    if (!node._AngeloCanvas) {
        attachPreviewCanvas(node);
    }
    const canvas = node._AngeloCanvas;
    const placeholder = node._AngeloPlaceholder;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
        const prev = node._AngeloImg;
        const sameDims = prev
            && prev.naturalWidth === img.naturalWidth
            && prev.naturalHeight === img.naturalHeight;
        node._AngeloImg = img;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        if (placeholder) placeholder.style.display = "none";

        // Tell Python the actual image dimensions so it can derive the
        // correct pixel→latent scale per-axis (FLUX 2 is 16×, FLUX 1
        // / SDXL are 8×; we don't want to hardcode either).
        const wi = findWidget(node, "image_w");
        const hi = findWidget(node, "image_h");
        if (wi) setWidget(wi, img.naturalWidth);
        if (hi) setWidget(hi, img.naturalHeight);

        // If this is a refine of the SAME image (same dims) and the user
        // is zoomed in, keep their view — so clicking to refine a detail
        // doesn't pop them back to fit. A genuinely new image (different
        // dims) or the first load resets to fit.
        if (sameDims && _angeloIsZoomed(node)) {
            applyView(node);
        } else {
            resetView(node);
        }

        // If we're in detect mode (candidates persist for batch editing),
        // re-overlay them on the freshly-refined preview so they stay put.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            redrawCanvasWithOverlays(node);
        }

        // Force a node redraw so LiteGraph re-computes the canvas widget size.
        if (node.graph && node.graph.setDirtyCanvas) {
            node.graph.setDirtyCanvas(true, false);
        }
        dbg("image drawn", { w: img.naturalWidth, h: img.naturalHeight });
    };
    img.onerror = (e) => {
        dbg("image load error", e, url);
    };
    img.src = url;
}

/**
 * Redraw the canvas with the underlying image + any active overlays
 * (hover ring, paint stroke). Called on every mousemove + on demand
 * from the paint handlers. Cheap: ~1 image blit + a few arcs per frame.
 */
function redrawCanvasWithOverlays(node) {
    const canvas = node._AngeloCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Before/after compare ('\' held): show the BASE image with a badge,
    // skip every overlay (read-only peek — no rings / strokes / detections).
    if (node._AngeloCompareHold && node._AngeloSourceImg) {
        ctx.drawImage(node._AngeloSourceImg, 0, 0, canvas.width, canvas.height);
        const fs = Math.max(16, Math.round(canvas.width * 0.022));
        const pad = Math.round(fs * 0.55);
        ctx.save();
        ctx.font = `bold ${fs}px Arial, sans-serif`;
        const label = "BEFORE — release \\ for after";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0, 0, 0, 0.62)";
        ctx.fillRect(pad, pad, tw + pad * 2, fs + pad * 1.4);
        ctx.fillStyle = "rgba(255, 220, 80, 0.95)";
        ctx.textBaseline = "top";
        ctx.fillText(label, pad * 2, pad * 1.7);
        ctx.restore();
        return;
    }

    if (node._AngeloImg) {
        ctx.drawImage(node._AngeloImg, 0, 0);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Cursor changes by mode — only remaining visual indicator now
    // that the corner pills are gone.
    if (canvas) {
        if (isOutpaintMode(node)) {
            canvas.style.cursor = node._AngeloOutpaintHoverDir ? "pointer" : "default";
        } else if (isSmartGuidedInpaintMode(node)) {
            canvas.style.cursor = "default";  // no canvas interaction
        } else if (isSmartInpaintMode(node)) {
            canvas.style.cursor = "crosshair";
        } else {
            canvas.style.cursor = isPaintModeOn(node) ? "cell" : "crosshair";
        }
    }

    // Outpaint mode: a glowing band along the hovered edge previews the
    // extension direction (with the amount), then nothing else — no hover
    // ring, no strokes, no detections in this mode.
    if (isOutpaintMode(node)) {
        const dir = node._AngeloOutpaintHoverDir;
        if (dir && node._AngeloImg) {
            const W = canvas.width, H = canvas.height;
            const t = Math.max(24, Math.round(Math.min(W, H) * 0.12));
            let x = 0, y = 0, w = W, h = H, glyph = "➡";
            if (dir === "left") { w = t; glyph = "⬅"; }
            else if (dir === "right") { x = W - t; w = t; glyph = "➡"; }
            else if (dir === "up") { h = t; glyph = "⬆"; }
            else if (dir === "down") { y = H - t; h = t; glyph = "⬇"; }
            ctx.save();
            let grad;
            if (dir === "left") grad = ctx.createLinearGradient(x, 0, x + w, 0);
            else if (dir === "right") grad = ctx.createLinearGradient(x + w, 0, x, 0);
            else if (dir === "up") grad = ctx.createLinearGradient(0, y, 0, y + h);
            else grad = ctx.createLinearGradient(0, y + h, 0, y);
            grad.addColorStop(0, "rgba(120, 190, 235, 0.55)");
            grad.addColorStop(1, "rgba(120, 190, 235, 0.0)");
            ctx.fillStyle = grad;
            ctx.fillRect(x, y, w, h);
            const amtW = findWidget(node, "outpaint_amount");
            const amt = (amtW && amtW.value) || 256;
            const fs = Math.max(16, Math.round(Math.min(W, H) * 0.04));
            ctx.font = `bold ${fs}px Arial, sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.shadowColor = "rgba(0, 0, 0, 0.85)";
            ctx.shadowBlur = 6;
            ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
            ctx.fillText(`${glyph} +${amt}px`, x + w / 2, y + h / 2);
            ctx.restore();
        }
        // Protect-brush overlay: painted circles in red (these stay frozen
        // inside the overlap band), plus a red brush ring at the cursor
        // while it's in the interior (i.e. when a drag would paint).
        const prot = node._AngeloProtect;
        if (prot && prot.length) {
            ctx.save();
            ctx.fillStyle = "rgba(255, 90, 90, 0.30)";
            for (const [px2, py2, pr2] of prot) {
                ctx.beginPath();
                ctx.arc(px2, py2, pr2, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.restore();
        }
        if (!dir && node._AngeloHover && node._AngeloImg) {
            const rect2 = canvas.getBoundingClientRect();
            const sx = rect2.width > 0 ? canvas.width / rect2.width : 1;
            const sy = rect2.height > 0 ? canvas.height / rect2.height : 1;
            const radiusW = findWidget(node, "click_radius");
            ctx.save();
            ctx.strokeStyle = "rgba(255, 90, 90, 0.8)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(node._AngeloHover.x * sx, node._AngeloHover.y * sy,
                (radiusW && radiusW.value) || 96, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        }
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? canvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? canvas.height / rect.height : 1;

    const radiusWidget = findWidget(node, "click_radius");
    const radiusPixel = radiusWidget ? radiusWidget.value : 96;

    // 1. Paint stroke (if actively painting) — draw the union of brush
    //    circles so far at 50% opacity (the user wanted it visibly
    //    blue without fully occluding the underlying image).
    if (node._AngeloPainting && node._AngeloStroke?.length) {
        ctx.save();
        ctx.fillStyle = "rgba(80, 180, 255, 0.5)";
        for (const [px, py] of node._AngeloStroke) {
            ctx.beginPath();
            ctx.arc(px, py, radiusPixel, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    }

    // 1b. Smart Inpaint live rectangle while the user is dragging.
    if (node._AngeloDraggingRect) {
        const r = node._AngeloDraggingRect;
        const x = Math.min(r.x1, r.x2);
        const y = Math.min(r.y1, r.y2);
        const w = Math.abs(r.x2 - r.x1);
        const h = Math.abs(r.y2 - r.y1);
        ctx.save();
        ctx.fillStyle = "rgba(80, 180, 255, 0.35)";
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = "rgba(80, 180, 255, 1.0)";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.restore();
    }

    // 2. Hover ring at cursor position (skip during active paint or
    //    rect drag — those have their own active visuals; skip in
    //    Smart Inpaint mode entirely since the click_radius circle
    //    is irrelevant to a rectangle workflow).
    if (node._AngeloHover
        && !node._AngeloPainting
        && !node._AngeloDraggingRect
        && !isSmartInpaintMode(node)) {
        const px = node._AngeloHover.x * scaleX;
        const py = node._AngeloHover.y * scaleY;
        ctx.save();
        ctx.strokeStyle = "rgba(255, 200, 80, 0.7)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, radiusPixel, 0, Math.PI * 2);
        ctx.stroke();
        // Tiny cross-hair at the centre
        ctx.strokeStyle = "rgba(255, 200, 80, 0.9)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px - 6, py); ctx.lineTo(px + 6, py);
        ctx.moveTo(px, py - 6); ctx.lineTo(px, py + 6);
        ctx.stroke();
        ctx.restore();
    }

    // 3. Detection candidates (SAM 3 / YOLO) awaiting a click-to-confirm.
    drawDetections(node, ctx);
}

function flashClickOverlay(node, cx, cy) {
    // Draw a fading ring on the canvas at the click point.
    // We don't have the click_radius -> pixel scaling on hand here cheaply;
    // just draw a 24px ring as visual feedback that the click registered.
    const canvas = node._AngeloCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    // Scale the click point from display coords to canvas coords (canvas
    // is at natural image size; display is css-scaled).
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const px = cx * scaleX;
    const py = cy * scaleY;

    const radiusWidget = findWidget(node, "click_radius");
    const radiusPixel = radiusWidget ? radiusWidget.value : 96;

    let t0 = performance.now();
    const tick = (t) => {
        const elapsed = t - t0;
        const alpha = Math.max(0, 1 - elapsed / 1500);
        if (alpha <= 0) return;
        // Redraw the underlying image then the ring on top
        if (node._AngeloImg) {
            ctx.drawImage(node._AngeloImg, 0, 0);
        }
        ctx.save();
        ctx.strokeStyle = `rgba(255, 200, 80, ${alpha})`;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(px, py, radiusPixel, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(255, 220, 120, ${alpha})`;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}


// ============================================================
// Refine trigger — update hidden widgets + queue
// ============================================================

function triggerRefine(node, pixelX, pixelY, displayCX, displayCY) {
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const ws = findWidget(node, "click_seq");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    if (!wx || !wy || !ws) {
        dbg("ERROR: hidden widgets not found", { wx: !!wx, wy: !!wy, ws: !!ws });
        return;
    }

    setWidget(wx, pixelX);
    setWidget(wy, pixelY);
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    if (wr) setWidget(wr, false);
    // Clear stroke_points + seg_polygon so a leftover paint stroke or a
    // SAM-detected silhouette from earlier doesn't bleed into a single
    // click refine. (seg_polygon persists across Persistent Mask re-rolls
    // until a manual action like this resets it.)
    if (wsp) setWidget(wsp, "");
    const wseg = findWidget(node, "seg_polygon");
    if (wseg) setWidget(wseg, "");
    const wmaskpng = findWidget(node, "seg_mask_png");
    if (wmaskpng) setWidget(wmaskpng, "");

    dbg("queueing workflow (click)", { click_x: wx.value, click_y: wy.value, click_seq: ws.value });
    queuePrompt(node);
}

function triggerPaintRefine(node, strokePoints) {
    const wsp = findWidget(node, "stroke_points");
    const ws = findWidget(node, "click_seq");
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    if (!wsp || !ws) {
        dbg("ERROR: stroke widgets not found", { wsp: !!wsp, ws: !!ws });
        return;
    }

    // Round to ints to keep the JSON tight. Sub-pixel precision is
    // wasted — Python rasterises into latent space anyway.
    const compact = strokePoints.map(([x, y]) => [Math.round(x), Math.round(y)]);
    setWidget(wsp, JSON.stringify(compact));
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // Also set click_x/y to the first stroke point as a fallback target,
    // in case anything downstream reads them.
    if (wx) setWidget(wx, compact[0][0]);
    if (wy) setWidget(wy, compact[0][1]);
    if (wr) setWidget(wr, false);
    // A paint stroke replaces any SAM-detected silhouette.
    const wseg = findWidget(node, "seg_polygon");
    if (wseg) setWidget(wseg, "");
    const wmaskpng = findWidget(node, "seg_mask_png");
    if (wmaskpng) setWidget(wmaskpng, "");

    dbg("queueing workflow (paint)", { points: compact.length, click_seq: ws.value });
    queuePrompt(node);
}

// Queue the workflow. Pass the initiating node for Angelo-internal action
// queues (a click/undo/re-roll/… on ONE node) — the persistent-mask queue
// hook then leaves every OTHER Angelo node alone, so editing node A can't
// re-run node B's held mask. Call with NO argument for a "real" Queue
// press (the in-node ▶ Queue button), which should bump all holders just
// like ComfyUI's own button.
function queuePrompt(node) {
    if (typeof app.queuePrompt === "function") {
        app._AngeloQueueInitiator = (node && node.id != null) ? node.id : null;
        try {
            const ret = app.queuePrompt(0);
            if (ret && typeof ret.then === "function") {
                ret.catch(e => dbg("queuePrompt promise rejected", e));
            }
        } finally {
            // The hook reads the flag synchronously during app.queuePrompt,
            // so it's safe (and important) to clear immediately after.
            app._AngeloQueueInitiator = null;
        }
    } else {
        dbg("ERROR: app.queuePrompt is not a function");
    }
}

// =====================================================================
// Load Image — bring an external photo in as the base latent.
// Button → file picker → resolution popup → upload (/upload/image) →
// set hidden widgets → queue. run() VAE-encodes the upload into the
// base; Reset/Undo then return to it.
// =====================================================================

// Persistent in-app notice bar (top of the preview). For actionable
// messages that must be read — unlike _angeloToast which auto-hides.
function showAngeloNotice(node, message) {
    if (!node._AngeloNotice) return;
    node._AngeloNoticeText.textContent = message;
    node._AngeloNotice.style.display = "block";
}
function hideAngeloNotice(node) {
    if (node._AngeloNotice) node._AngeloNotice.style.display = "none";
}

function showAngeloLoading(node, message) {
    if (!node._AngeloLoading) return;
    if (message) node._AngeloLoadingText.textContent = message;
    node._AngeloLoading.style.display = "flex";
}
function hideAngeloLoading(node) {
    if (node._AngeloLoading) node._AngeloLoading.style.display = "none";
}

function _angeloToast(message) {
    const t = document.createElement("div");
    t.textContent = message;
    t.style.cssText = "position:fixed; top:20px; right:20px; background:#333; color:#fff; "
        + "padding:10px 16px; border-radius:6px; z-index:100000; font:13px Arial,sans-serif; "
        + "box-shadow:0 2px 8px rgba(0,0,0,0.5); opacity:0; transition:opacity 0.18s;";
    document.body.appendChild(t);
    requestAnimationFrame(() => { t.style.opacity = "1"; });
    setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 220); }, 1600);
}

function triggerLoadImage(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.style.display = "none";
    input.addEventListener("change", () => {
        const file = input.files && input.files[0];
        if (file) showLoadImagePopup(node, file);
        input.remove();
    });
    document.body.appendChild(input);
    input.click();
}

// ---- Right-click image actions (#7) -------------------------------------
// Copy / Open, like ComfyUI's image nodes. Loading an image is via drag-drop
// onto the node, the Load Image button, or Ctrl+V / Cmd+V while hovering
// the preview (the window-level paste handler near the top of this file).

function _angeloOpenImageInTab(node) {
    let url = null;
    if (node._AngeloPreviewRef) url = makeViewUrl(node._AngeloPreviewRef);
    else if (node._AngeloImg && node._AngeloImg.src) url = node._AngeloImg.src;
    if (url) window.open(url, "_blank");
    else _angeloToast("No image to open yet");
}

async function _angeloCopyImageToClipboard(node) {
    const img = node._AngeloImg;
    if (!img) { _angeloToast("No image to copy yet"); return; }
    if (!navigator.clipboard || !window.ClipboardItem) {
        _angeloToast("Clipboard image copy not supported in this browser");
        return;
    }
    try {
        const c = document.createElement("canvas");
        c.width = img.naturalWidth || img.width;
        c.height = img.naturalHeight || img.height;
        c.getContext("2d").drawImage(img, 0, 0);
        const blob = await new Promise((res) => c.toBlob(res, "image/png"));
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        _angeloToast("Image copied to clipboard");
    } catch (e) {
        dbg("[Angelo] copy image failed", e);
        _angeloToast("Copy failed — see console");
    }
}

// ===== Right-click quick actions (#29) ==============================
// The generate→edit loop without hunting for the toolbar: mode switch +
// the two base actions live on BOTH right-click menus (node body via
// getExtraMenuOptions, preview canvas via _angeloShowImageContextMenu —
// the canvas one is what a user zoomed into the preview actually reaches).
// "New base" vs "same base" is the distinction #29 asked for explicitly.

function _angeloSetMode(node, mode) {
    // Same setWidget path as the Mode dropdown, so the wrapped mode
    // callback fires (seed lock/restore + toolbar grey sync) — then
    // mirror the DOM dropdown, which only the dropdown's own click
    // handler would otherwise update.
    const w = findWidget(node, "mode");
    if (!w || String(w.value) === mode) return;
    setWidget(w, mode);
    syncModeSelect(node);
}

function _angeloGenerateNewBase(node) {
    // One gesture for "start the cycle again": Sampler Mode + a fresh
    // random seed + queue. The seed roll is unconditional — the user
    // asked for a NEW base — but the control widget is left alone, so
    // a deliberate `fixed` stays fixed (at the new seed) afterwards.
    _angeloSetMode(node, "Sampler Mode");
    const seedW = findWidget(node, "sampler_seed");
    if (seedW) {
        setWidget(seedW, Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
        syncSamplerSeedInput(node);
    }
    queuePrompt(node);
}

function _angeloRegenerateSameBase(node) {
    // Re-run the exact base generation: Sampler Mode + the seed Python
    // actually used for the cached base (seed_at_run — the widget value
    // may have drifted via after-gen control). Control is untouched:
    // randomize users get their roll back after this run as usual.
    _angeloSetMode(node, "Sampler Mode");
    const seedW = findWidget(node, "sampler_seed");
    const stored = node._AngeloSamplerSeedAtRun;
    if (seedW && stored != null && Number(seedW.value) !== Number(stored)) {
        setWidget(seedW, Number(stored));
        syncSamplerSeedInput(node);
    }
    queuePrompt(node);
}

function _angeloShowImageContextMenu(node, event) {
    const hasImg = !!node._AngeloImg;
    const inSampler = String(findWidget(node, "mode")?.value) === "Sampler Mode";
    const items = [
        { content: inSampler ? "Switch to Edit Mode" : "Switch to Sampler Mode",
          callback: () => _angeloSetMode(node, inSampler ? "Edit Mode" : "Sampler Mode") },
        { content: "Generate new base (fresh seed)",
          callback: () => _angeloGenerateNewBase(node) },
        { content: "Regenerate same base",
          callback: () => _angeloRegenerateSameBase(node) },
        null,
        { content: "Open image in new tab", disabled: !hasImg,
          callback: () => _angeloOpenImageInTab(node) },
        { content: "Copy image", disabled: !hasImg,
          callback: () => _angeloCopyImageToClipboard(node) },
    ];
    const LG = window.LiteGraph;
    if (LG && LG.ContextMenu) {
        const menu = new LG.ContextMenu(items, { event, title: "Angelo" });
        // In fullscreen the overlay sits at z-index 99990 — above the stock
        // litecontextmenu z-index — so lift the menu root or it renders
        // invisibly behind the overlay.
        if (node._AngeloFSOverlay && menu && menu.root) {
            menu.root.style.zIndex = "100001";
        }
    }
}

async function _uploadLoadedImage(node, file, mode, mp) {
    const fd = new FormData();
    fd.append("image", file, file.name);
    fd.append("overwrite", "false");
    _angeloToast("Uploading image…");
    let data;
    try {
        const res = await api.fetchApi("/upload/image", { method: "POST", body: fd });
        if (!res.ok) { _angeloToast("Upload failed"); return; }
        data = await res.json();
    } catch (e) {
        dbg("[Angelo] upload failed", e);
        _angeloToast("Upload failed — see console");
        return;
    }
    const ref = JSON.stringify({
        name: data.name,
        subfolder: data.subfolder || "",
        type: data.type || "input",
    });
    const wImg = findWidget(node, "loaded_image");
    const wMode = findWidget(node, "loaded_resize_mode");
    const wMp = findWidget(node, "loaded_target_mp");
    const wSeq = findWidget(node, "loaded_image_seq");
    if (wImg) setWidget(wImg, ref);
    if (wMode) setWidget(wMode, mode);
    if (wMp) setWidget(wMp, mp);
    if (wSeq) setWidget(wSeq, ((wSeq.value || 0) + 1) & 0x7FFFFFFF);

    // Loading a photo means you want to EDIT it — flip to Edit Mode so
    // the queue refines/previews the loaded base instead of regenerating
    // it from noise (Sampler Mode at denoise 1.0). Fires the wrapped mode
    // callback (lock + grey sync); refresh the Mode dropdown to match.
    const wNodeMode = findWidget(node, "mode");
    if (wNodeMode && String(wNodeMode.value) !== "Edit Mode") {
        setWidget(wNodeMode, "Edit Mode");
        syncModeSelect(node);
    }

    syncLoadImageControls(node);   // reveal the Unload button
    _angeloToast("Loading as base…");
    queuePrompt(node);
}

// Clear the loaded image → the wired latent input takes over as base.
function unloadImage(node) {
    const wImg = findWidget(node, "loaded_image");
    if (wImg) setWidget(wImg, "");
    syncLoadImageControls(node);
    _angeloToast("Unloaded — using latent input");
    queuePrompt(node);
}

// Show the Unload button only while an image is loaded.
function syncLoadImageControls(node) {
    const btn = node._AngeloUnloadImageBtn;
    if (!btn) return;
    const w = findWidget(node, "loaded_image");
    const active = !!(w && String(w.value || "").trim());
    btn.style.display = active ? "" : "none";
}

// =====================================================================
// Detect (SAM 3 auto-segment): text → /angelo/detect → highlighted
// candidates on the canvas → click-to-confirm → mask per mode → refine.
// Refine uses the silhouette polygons; Smart Inpaint uses the bbox.
// =====================================================================

async function runDetect(node, conceptOverride) {
    // conceptOverride (from a quick-select preset) runs that term directly
    // WITHOUT touching the text box; otherwise use what the user typed.
    const text = (conceptOverride != null && String(conceptOverride).trim())
        ? String(conceptOverride).trim()
        : (node._AngeloDetectText?.value || "").trim();
    if (!text) { showAngeloNotice(node, "Type what to segment first (e.g. \"the face\")."); return; }
    const ref = node._AngeloPreviewRef;
    if (!ref || !ref.filename) { showAngeloNotice(node, "Generate or load an image first, then Detect."); return; }
    const confEl = node._AngeloDetectConf && node._AngeloDetectConf._AngeloInput;
    const conf = confEl ? Math.max(0.05, Math.min(0.95, parseFloat(confEl.value) || 0.3)) : 0.3;
    hideAngeloNotice(node);   // clear any prior error before a new attempt
    // In-app overlay (NOT a toast) while the request is in flight — the
    // first detect builds the SAM 3 model and can take several seconds.
    showAngeloLoading(node, "Loading SAM 3…");
    try {
        const res = await api.fetchApi("/angelo/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                method: "sam3_text",
                text,
                confidence_threshold: conf,
                filename: ref.filename,
                subfolder: ref.subfolder || "",
                type: ref.type || "temp",
            }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            dbg("[Angelo] detect error", data);
            // Persistent in-app notice — esp. the "SAM 3 not installed →
            // run the installer" message, which needs reading + acting on.
            showAngeloNotice(node, data.error || `Detect failed (HTTP ${res.status}).`);
            return;
        }
        const dets = data.detections || [];
        node._AngeloDetections = dets;
        node._AngeloHoverDet = -1;
        node._AngeloEditedDets = new Set();   // which candidates have been edited
        node._AngeloMaskGrow = 0;             // each detect starts at no grow
        _syncMaskGrowReadout(node);
        syncDetectModeButton(node);
        if (!dets.length) {
            // Nothing highlights, so say so where it can be read + acted on.
            showAngeloNotice(node, "No matches — try a lower Conf or different words.");
        }
        redrawCanvasWithOverlays(node);
    } catch (e) {
        dbg("[Angelo] detect failed", e);
        showAngeloNotice(node, "Detect request failed — is the ComfyUI server reachable? See the console.");
    } finally {
        hideAngeloLoading(node);   // self-dismiss on every exit path
    }
}

// ===== Fix All — sequential auto-edit of every remaining candidate =====
// State: node._AngeloFixAll = { queue: [det indices], total }. The loop is
// event-driven: confirmDetection queues a run; when that run's onExecuted
// arrives, _angeloFixAllStep fires the next candidate. Stopping (button,
// Cancel Detect, Esc) clears the state — the in-flight candidate still
// lands, nothing after it fires.

function toggleFixAll(node) {
    if (node._AngeloFixAll) {
        _angeloFixAllFinish(node, "stopped");
        return;
    }
    const dets = node._AngeloDetections || [];
    const edited = node._AngeloEditedDets || new Set();
    const queue = dets.map((_, i) => i).filter((i) => !edited.has(i));
    if (!queue.length) {
        _angeloToast("Nothing left to fix — every candidate is already done");
        return;
    }
    node._AngeloFixAll = { queue, total: queue.length };
    _angeloFixAllUpdateButton(node);
    _angeloFixAllStep(node);
}

function _angeloFixAllStep(node) {
    const fa = node._AngeloFixAll;
    if (!fa) return;
    const dets = node._AngeloDetections;
    if (!dets || !dets.length || !fa.queue.length) {
        _angeloFixAllFinish(node, "complete");
        return;
    }
    const idx = fa.queue.shift();
    const det = dets[idx];
    if (!det) {
        _angeloFixAllStep(node);
        return;
    }
    _angeloFixAllUpdateButton(node);
    confirmDetection(node, det);
}

function _angeloFixAllUpdateButton(node) {
    const btn = node._AngeloFixAllBtn;
    if (!btn) return;
    const fa = node._AngeloFixAll;
    if (!fa) {
        btn.textContent = "⚡ Fix All";
        btn.style.background = "rgba(30,120,80,0.95)";
        btn.style.borderColor = "#4a7";
        return;
    }
    const done = fa.total - fa.queue.length;
    btn.textContent = `■ Stop (${done}/${fa.total})`;
    btn.style.background = "rgba(150,60,40,0.95)";
    btn.style.borderColor = "#e96";
}

function _angeloFixAllFinish(node, why) {
    if (!node._AngeloFixAll) return;
    node._AngeloFixAll = null;
    _angeloFixAllUpdateButton(node);
    _angeloToast(why === "complete" ? "Fix All complete ✓" : "Fix All stopped");
}

function clearDetections(node) {
    if (!node._AngeloDetections) return;
    // Leaving detect mode aborts a running Fix All (silently — the toast
    // would be noise when the user explicitly cancelled).
    if (node._AngeloFixAll) {
        node._AngeloFixAll = null;
        _angeloFixAllUpdateButton(node);
    }
    node._AngeloDetections = null;   // candidate objects (+ their _editMask) drop here
    node._AngeloHoverDet = -1;
    node._AngeloEditedDets = null;
    node._AngeloTouchup = null;
    node._AngeloBrushPreview = null;
    // Reset the highlight opacity to default so the next detect starts full.
    node._AngeloDetOpacity = 1.0;
    if (node._AngeloDetOpacitySlider) node._AngeloDetOpacitySlider.value = "1";
    node._AngeloMaskGrow = 0;
    _syncMaskGrowReadout(node);
    syncDetectModeButton(node);
    redrawCanvasWithOverlays(node);
}

// Show the floating detect-mode panel (Cancel + opacity slider) while
// candidates are active.
function syncDetectModeButton(node) {
    const panel = node._AngeloDetectPanel;
    if (!panel) return;
    panel.style.display = (node._AngeloDetections && node._AngeloDetections.length) ? "flex" : "none";
}

// Topmost (tightest) detection whose bbox contains the image-pixel point.
function _detAtPoint(node, px, py) {
    const dets = node._AngeloDetections || [];
    let best = null, bestArea = Infinity;
    for (const d of dets) {
        const b = _detBbox(node, d);
        if (!b) continue;
        if (px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3]) {
            const area = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
            if (area < bestArea) { bestArea = area; best = d; }
        }
    }
    return best;
}

function confirmDetection(node, det) {
    if (isSmartGuidedInpaintMode(node)) return;
    const ws = findWidget(node, "click_seq");
    if (!ws) return;
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    const wrp = findWidget(node, "rect_points");
    const wseg = findWidget(node, "seg_polygon");
    const wmask = findWidget(node, "seg_mask_png");
    const smart = isSmartInpaintMode(node);
    // Apply the current Mask grow/shrink to the committed shape (the same
    // offset that's being drawn), so what you edit matches what you see.
    const b = _detBbox(node, det) || [0, 0, 0, 0];
    if (wx) setWidget(wx, Math.round((b[0] + b[2]) / 2));
    if (wy) setWidget(wy, Math.round((b[1] + b[3]) / 2));
    if (wr) setWidget(wr, false);
    if (wsp) setWidget(wsp, "");
    if (smart) {
        // Smart Inpaint: the (grown) bbox is the rectangle. (No touch-up
        // brush in Smart Inpaint — it's Refine-only.)
        if (wrp) setWidget(wrp, JSON.stringify([[
            Math.round(b[0]), Math.round(b[1]), Math.round(b[2]), Math.round(b[3]),
        ]]));
        if (wseg) setWidget(wseg, "");
        if (wmask) setWidget(wmask, "");
    } else if (det._editMask && wmask) {
        // Refine + brushed: send the raster edit-mask (handles brushed holes /
        // additions a polygon can't), and clear the polygon path.
        const png = det._editMask.toDataURL("image/png").split(",")[1] || "";
        setWidget(wmask, png);
        if (wseg) setWidget(wseg, "");
        if (wrp) setWidget(wrp, "");
    } else {
        // Refine: the (grown) silhouette polygons are the mask.
        if (wseg) setWidget(wseg, JSON.stringify(_detPolys(node, det)));
        if (wmask) setWidget(wmask, "");
        if (wrp) setWidget(wrp, "");
    }
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // Keep the candidates up (batch editing): mark this one edited (drawn
    // green) and leave the rest clickable. Exit via Cancel / Esc / Space.
    const idx = node._AngeloDetections ? node._AngeloDetections.indexOf(det) : -1;
    if (idx >= 0) {
        node._AngeloEditedDets = node._AngeloEditedDets || new Set();
        node._AngeloEditedDets.add(idx);
    }
    redrawCanvasWithOverlays(node);   // immediate feedback: clicked one turns green
    queuePrompt(node);
}

// Offset a closed polygon outward (delta>0) or inward (delta<0) by ~|delta|
// pixels, perpendicular to its edges. Miter join with a spike clamp; the
// outward direction is fixed per-vertex via the centroid so winding order
// doesn't matter. flat = [x0,y0,x1,y1,...] in image-pixel coords.
function _offsetPolygon(flat, delta) {
    const n = flat.length >> 1;
    if (n < 3 || !delta) return flat.slice();
    const P = [];
    for (let i = 0; i < n; i++) P.push([flat[2 * i], flat[2 * i + 1]]);
    let cx = 0, cy = 0;
    for (const [x, y] of P) { cx += x; cy += y; }
    cx /= n; cy /= n;
    const out = new Array(n * 2);
    for (let i = 0; i < n; i++) {
        const prev = P[(i - 1 + n) % n], cur = P[i], next = P[(i + 1) % n];
        let e1x = cur[0] - prev[0], e1y = cur[1] - prev[1];
        let e2x = next[0] - cur[0], e2y = next[1] - cur[1];
        const l1 = Math.hypot(e1x, e1y) || 1, l2 = Math.hypot(e2x, e2y) || 1;
        e1x /= l1; e1y /= l1; e2x /= l2; e2y /= l2;
        // Unit edge normals (one consistent side); miter = their sum.
        const n1x = -e1y, n1y = e1x, n2x = -e2y, n2y = e2x;
        let mx = n1x + n2x, my = n1y + n2y;
        let ml = Math.hypot(mx, my);
        if (ml < 1e-6) { mx = n1x; my = n1y; ml = Math.hypot(mx, my) || 1; }
        mx /= ml; my /= ml;
        let cosA = Math.abs(mx * n1x + my * n1y);   // cos(half-angle)
        if (cosA < 0.25) cosA = 0.25;               // clamp miter spike at sharp corners
        const len = Math.abs(delta) / cosA;
        // Orient the miter outward (away from centroid), then push out/in.
        if (mx * (cur[0] - cx) + my * (cur[1] - cy) < 0) { mx = -mx; my = -my; }
        const s = delta >= 0 ? 1 : -1;
        out[2 * i] = cur[0] + mx * len * s;
        out[2 * i + 1] = cur[1] + my * len * s;
    }
    return out;
}

// The detection's polygons / bbox AS DISPLAYED + COMMITTED, with the
// current mask grow/shrink applied. grow==0 returns the originals.
function _detPolys(node, det) {
    const g = node._AngeloMaskGrow || 0;
    const polys = det.polygons || [];
    if (!g) return polys;
    return polys.map((p) => (p && p.length >= 6) ? _offsetPolygon(p, g) : p);
}
function _detBbox(node, det) {
    if (!det.bbox) return det.bbox;
    const g = node._AngeloMaskGrow || 0;
    let b = det.bbox;
    if (g) {
        const img = node._AngeloImg;
        const W = (img && img.naturalWidth) ? img.naturalWidth : 1e9;
        const H = (img && img.naturalHeight) ? img.naturalHeight : 1e9;
        b = [
            Math.max(0, b[0] - g), Math.max(0, b[1] - g),
            Math.min(W, b[2] + g), Math.min(H, b[3] + g),
        ];
    }
    // Include any brushed extent so the whole touched shape is hit-testable.
    const e = det._editBounds;
    if (e) {
        b = [Math.min(b[0], e[0]), Math.min(b[1], e[1]), Math.max(b[2], e[2]), Math.max(b[3], e[3])];
    }
    return b;
}

function _syncMaskGrowReadout(node) {
    const r = node._AngeloMaskGrowReadout;
    if (!r) return;
    const g = node._AngeloMaskGrow || 0;
    r.textContent = (g > 0 ? "+" : "") + g + "px";
}

// +/- buttons: nudge the grow value and re-draw every highlight together.
function adjustMaskGrow(node, delta) {
    if (!node._AngeloDetections || !node._AngeloDetections.length) return;
    const cur = node._AngeloMaskGrow || 0;
    node._AngeloMaskGrow = Math.max(-40, Math.min(200, cur + delta));
    _syncMaskGrowReadout(node);
    redrawCanvasWithOverlays(node);
}

// ===== Touch-up brush (Refine only) — Shift-drag adds to a detected mask,
//       Alt-drag subtracts (holes allowed). Once brushed, a detection holds
//       a raster edit-mask (`_editMask`, an offscreen canvas at image res,
//       white = masked) that is its source of truth for display + commit. =====

function _brushRadius(node) {
    const w = findWidget(node, "click_radius");
    return Math.max(2, (w && Number(w.value)) || 96);
}

// Lazily promote a detection to a raster edit-mask, seeded from its current
// (grown) silhouette so brushing starts from exactly what's on screen.
function _ensureEditMask(node, det) {
    if (det._editMask) return det._editMask;
    const img = node._AngeloImg;
    const W = (img && img.naturalWidth) || 512;
    const H = (img && img.naturalHeight) || 512;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const cx = c.getContext("2d");
    cx.fillStyle = "#fff";
    for (const poly of _detPolys(node, det)) {
        if (!poly || poly.length < 6) continue;
        cx.beginPath();
        cx.moveTo(poly[0], poly[1]);
        for (let k = 2; k < poly.length - 1; k += 2) cx.lineTo(poly[k], poly[k + 1]);
        cx.closePath();
        cx.fill();
    }
    det._editMask = c;
    return c;
}

function _brushStamp(det, px, py, radius, subtract) {
    const cx = det._editMask.getContext("2d");
    cx.save();
    cx.globalCompositeOperation = subtract ? "destination-out" : "source-over";
    cx.fillStyle = "#fff";
    cx.beginPath();
    cx.arc(px, py, Math.max(1, radius), 0, Math.PI * 2);
    cx.fill();
    cx.restore();
    // Grow the hit-box to cover added paint so the whole brushed shape stays
    // hoverable / clickable even where it extends past the original bbox.
    if (!subtract) {
        const nb = [px - radius, py - radius, px + radius, py + radius];
        const b = det._editBounds;
        det._editBounds = b
            ? [Math.min(b[0], nb[0]), Math.min(b[1], nb[1]), Math.max(b[2], nb[2]), Math.max(b[3], nb[3])]
            : nb;
    }
}

// Stamp circles along a segment so a fast drag leaves a continuous stroke.
function _brushLine(det, a, b, radius, subtract) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const dist = Math.hypot(dx, dy);
    const n = Math.max(1, Math.ceil(dist / Math.max(1, radius * 0.4)));
    for (let i = 1; i <= n; i++) {
        const t = i / n;
        _brushStamp(det, a[0] + dx * t, a[1] + dy * t, radius, subtract);
    }
}

// Which candidate a brush stroke acts on: the one under the cursor (smallest
// bbox), else the nearest by bbox-centre so a stroke starting just outside a
// mask still attaches to it.
function _pickTouchupTarget(node, px, py) {
    const dets = node._AngeloDetections || [];
    if (!dets.length) return null;
    const hit = _detAtPoint(node, px, py);
    if (hit) return hit;
    let best = null, bestD = Infinity;
    for (const d of dets) {
        const b = _detBbox(node, d);
        if (!b) continue;
        const dxc = (b[0] + b[2]) / 2 - px, dyc = (b[1] + b[3]) / 2 - py;
        const dist = Math.hypot(dxc, dyc);
        if (dist < bestD) { bestD = dist; best = d; }
    }
    return best;
}

// Reusable scratch canvas for tinting a raster edit-mask in a candidate's
// colour before compositing it onto the overlay.
let _angeloScratch = null;
function _getScratch(w, h) {
    if (!_angeloScratch) _angeloScratch = document.createElement("canvas");
    if (_angeloScratch.width !== w) _angeloScratch.width = w;
    if (_angeloScratch.height !== h) _angeloScratch.height = h;
    return _angeloScratch;
}
function _drawTintedMask(ctx, det, color, fillAlpha) {
    const m = det._editMask;
    const W = m.width, H = m.height;
    const s = _getScratch(W, H);
    const sx = s.getContext("2d");
    sx.clearRect(0, 0, W, H);
    sx.globalCompositeOperation = "source-over";
    sx.fillStyle = color;
    sx.fillRect(0, 0, W, H);
    sx.globalCompositeOperation = "destination-in";
    sx.drawImage(m, 0, 0);
    const prev = ctx.globalAlpha;
    ctx.globalAlpha = prev * fillAlpha;
    ctx.drawImage(s, 0, 0, W, H);
    ctx.globalAlpha = prev;
}

// Draw candidate outlines (called from redrawCanvasWithOverlays, image-px ctx).
function drawDetections(node, ctx) {
    const dets = node._AngeloDetections;
    if (!dets || !dets.length) return;
    const smart = isSmartInpaintMode(node);
    ctx.save();
    // Selection-highlight opacity (the floating slider) scales the whole
    // overlay so the user can dim it and inspect the edited region's edges.
    ctx.globalAlpha = (typeof node._AngeloDetOpacity === "number") ? node._AngeloDetOpacity : 1;
    const edited = node._AngeloEditedDets;
    dets.forEach((d, i) => {
        const hot = (i === node._AngeloHoverDet);
        const done = edited && edited.has(i);
        let tint;   // solid colour for the raster-mask fill
        if (hot) {
            ctx.lineWidth = 4;
            ctx.strokeStyle = "rgba(255, 220, 80, 1.0)";
            ctx.fillStyle = "rgba(255, 220, 80, 0.25)";
            tint = "rgb(255, 220, 80)";
        } else if (done) {
            // already edited this session — green so the user can track progress
            ctx.lineWidth = 2;
            ctx.strokeStyle = "rgba(90, 220, 120, 0.95)";
            ctx.fillStyle = "rgba(90, 220, 120, 0.18)";
            tint = "rgb(90, 220, 120)";
        } else {
            ctx.lineWidth = 2;
            ctx.strokeStyle = "rgba(80, 200, 255, 0.9)";
            ctx.fillStyle = "rgba(80, 200, 255, 0.15)";
            tint = "rgb(80, 200, 255)";
        }
        if (!smart && d._editMask) {
            // Brushed candidate: its raster edit-mask is the source of truth.
            // A raster has no outline to thicken on hover like the polygon
            // candidates, so instead brighten the fill + add a glow so the
            // hovered one still pops clearly.
            if (hot) {
                ctx.save();
                ctx.shadowColor = tint;
                ctx.shadowBlur = 16;
                _drawTintedMask(ctx, d, tint, 0.55);
                ctx.restore();
            } else {
                _drawTintedMask(ctx, d, tint, 0.30);
            }
        } else if (smart) {
            const b = _detBbox(node, d);
            if (b) {
                ctx.beginPath();
                ctx.rect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
                ctx.fill(); ctx.stroke();
            }
        } else {
            for (const poly of _detPolys(node, d)) {
                if (!poly || poly.length < 6) continue;
                ctx.beginPath();
                ctx.moveTo(poly[0], poly[1]);
                for (let k = 2; k < poly.length - 1; k += 2) ctx.lineTo(poly[k], poly[k + 1]);
                ctx.closePath();
                ctx.fill(); ctx.stroke();
            }
        }
    });

    // Touch-up brush preview — a circle at the cursor while Shift/Alt is held
    // (green = add, red = subtract).
    const bp = node._AngeloBrushPreview;
    if (bp) {
        ctx.globalAlpha = (typeof node._AngeloDetOpacity === "number") ? node._AngeloDetOpacity : 1;
        ctx.lineWidth = 2;
        ctx.strokeStyle = bp.subtract ? "rgba(255, 90, 90, 0.95)" : "rgba(90, 230, 130, 0.95)";
        ctx.fillStyle = bp.subtract ? "rgba(255, 90, 90, 0.12)" : "rgba(90, 230, 130, 0.12)";
        ctx.beginPath();
        ctx.arc(bp.x, bp.y, bp.r, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
    }
    ctx.restore();
}

// Show the Detect row only in Edit Mode's masked sub-modes (Refine +
// Smart Inpaint); hide in Smart Guided (no mask) and Sampler Mode.
function syncDetectControls(node) {
    const row = node._AngeloDetectRow;
    if (!row) return;
    const modeW = findWidget(node, "mode");
    const inEdit = modeW && String(modeW.value) === "Edit Mode";
    const show = inEdit && !isSmartGuidedInpaintMode(node) && !isOutpaintMode(node);
    // Must restore "flex" (not "") — an empty string reverts the row to a
    // <div>'s default display:block, which kills flex-wrap:nowrap and the
    // separator's align-self:stretch (dropdown drops to a new line, sep
    // vanishes).
    row.style.display = show ? "flex" : "none";
    if (!show) clearDetections(node);
}

function showLoadImagePopup(node, file) {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed; inset:0; background:rgba(0,0,0,0.6); "
        + "display:flex; align-items:center; justify-content:center; z-index:10000; "
        + "font-family:Arial,sans-serif;";

    const modal = document.createElement("div");
    modal.style.cssText = "background:#2a2a2a; color:#ddd; border:1px solid #555; "
        + "border-radius:8px; padding:16px; width:360px; max-width:90vw; "
        + "display:flex; flex-direction:column; gap:10px;";

    const header = document.createElement("div");
    header.textContent = "Load Image — resolution";
    header.style.cssText = "font-size:14px; font-weight:bold; color:#aaa;";
    modal.appendChild(header);

    const hint = document.createElement("div");
    hint.textContent = `"${file.name}" — both options round dimensions to a multiple of 16.`;
    hint.style.cssText = "font-size:11px; color:#888;";
    modal.appendChild(hint);

    // Two mutually-exclusive choices via radio inputs.
    const mkRadio = (value, labelText, checked) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex; align-items:center; gap:8px; font-size:13px; cursor:pointer; padding:3px 2px;";
        const r = document.createElement("input");
        r.type = "radio";
        r.name = "angelo_loadres";
        r.value = value;
        r.checked = !!checked;
        const span = document.createElement("span");
        span.textContent = labelText;
        row.appendChild(r);
        row.appendChild(span);
        return { row, radio: r };
    };
    const keep = mkRadio("keep", "Keep current resolution", true);
    const resize = mkRadio("mp", "Resize to", false);
    modal.appendChild(keep.row);

    // Resize row: radio + MP input + "MP" label, inline.
    const mpInput = document.createElement("input");
    mpInput.type = "number";
    mpInput.min = "0.1"; mpInput.max = "8"; mpInput.step = "0.1"; mpInput.value = "1.5";
    mpInput.style.cssText = "width:60px; background:#1a1a1a; color:#eee; border:1px solid #555; border-radius:3px; padding:2px 6px; font-size:12px;";
    const mpLabel = document.createElement("span");
    mpLabel.textContent = "MP";
    mpLabel.style.cssText = "font-size:13px;";
    resize.row.appendChild(mpInput);
    resize.row.appendChild(mpLabel);
    modal.appendChild(resize.row);
    // Picking the MP field implies the resize choice.
    mpInput.addEventListener("focus", () => { resize.radio.checked = true; });

    // Discoverability: the load dialog is where the restoration workflow
    // starts, so teach it here.
    const tip = document.createElement("div");
    tip.style.cssText = "margin-top:4px; padding:8px 10px; font-size:11.5px; line-height:1.5; "
        + "color:#ffe9b0; background:rgba(110,85,25,0.25); "
        + "border:1px solid rgba(240,200,90,0.45); border-radius:5px;";
    tip.innerHTML = "💡 <b>Low-quality or small photo?</b> Load it at <b>1.5–3 MP</b> with "
        + "“Resize to”, then press <b>✨ Quick Photo Refine</b> once it's in. The image is "
        + "rebuilt sharp at the new size, anchored to the original — same person, same scene, "
        + "real detail.";
    modal.appendChild(tip);

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex; justify-content:flex-end; gap:8px; margin-top:8px;";
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = "background:#444; color:#ddd; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;";
    const loadBtn = document.createElement("button");
    loadBtn.textContent = "Load";
    loadBtn.style.cssText = "background:rgba(30,120,80,0.95); color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;";
    footer.appendChild(cancelBtn);
    footer.appendChild(loadBtn);
    modal.appendChild(footer);

    backdrop.appendChild(modal);
    _angeloShowModal(node, backdrop);

    const close = () => { if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop); };
    cancelBtn.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
    loadBtn.addEventListener("click", () => {
        const mode = resize.radio.checked ? "mp" : "keep";
        const mp = Math.max(0.1, Math.min(8, parseFloat(mpInput.value) || 1.5));
        close();
        _uploadLoadedImage(node, file, mode, mp);
    });
}

function isPaintModeOn(node) {
    const w = findWidget(node, "paint_mode");
    if (!w) {
        dbg("isPaintModeOn: paint_mode widget NOT FOUND on node — widget list:",
            (node.widgets || []).map(x => x.name));
        return false;
    }
    const v = w.value;
    // Coerce defensively — different ComfyUI versions may store BOOLEAN
    // widgets as actual booleans, the strings "true"/"false", or 0/1.
    const on = (v === true || v === 1 || v === "true" || v === "True" || v === "1");
    return on;
}

function isSmartInpaintMode(node) {
    const w = findWidget(node, "inpainting_mode");
    if (!w) return false;
    return w.value === "Smart Inpaint";
}

function isSmartGuidedInpaintMode(node) {
    const w = findWidget(node, "inpainting_mode");
    if (!w) return false;
    return w.value === "Smart Guided Inpaint";
}

// Either edit-model mode (both inject reference_latents).
function isAnySmartMode(node) {
    return isSmartInpaintMode(node) || isSmartGuidedInpaintMode(node);
}

// Outpaint instruction texts — MUST match Python's _OUTPAINT_INSTRUCTIONS
// exactly (Python owns the encode; this mirror only feeds the live
// combined-prompt preview under the Area Prompt box).
const _Angelo_OUTPAINT_INSTRUCTIONS = {
    left:  "Extend the image to the left, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    right: "Extend the image to the right, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    up:    "Extend the image upward, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    down:  "Extend the image downward, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    all:   "Extend the image outward on all sides, continuing the scene and background naturally. Do not repeat or add new subjects. ",
};

// Smart Guided Inpaint location labels — MUST match the Python
// _GUIDED_LOCATION_PREFIXES keys exactly (Python owns the label→prefix
// mapping; JS only stores the chosen label).
const _Angelo_GUIDED_LOCATIONS = [
    "(none)", "Whole image",
    "Top left", "Top middle", "Top right",
    "Middle left", "Center", "Middle right",
    "Bottom left", "Bottom middle", "Bottom right",
    "Left edge", "Right edge", "Top edge", "Bottom edge",
    "Top half", "Bottom half", "Left half", "Right half",
    "Top of the image", "Bottom of the image",
];

// Smart Guided Inpaint has no click/drag — this fires the backend's
// new_click gate so the whole-image guided edit runs. Sets a valid
// (image-centre) click point, bumps click_seq, clears any stale
// stroke/rect, and queues.
function triggerGuidedRefine(node) {
    const ws = findWidget(node, "click_seq");
    if (!ws) return;
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    const wrp = findWidget(node, "rect_points");
    const img = node._AngeloImg;
    const cx = img && img.naturalWidth ? Math.round(img.naturalWidth / 2) : 0;
    const cy = img && img.naturalHeight ? Math.round(img.naturalHeight / 2) : 0;
    if (wx) setWidget(wx, cx);
    if (wy) setWidget(wy, cy);
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    if (wr) setWidget(wr, false);
    if (wsp) setWidget(wsp, "");
    if (wrp) setWidget(wrp, "");
    dbg("queueing workflow (smart guided edit)", { click_seq: ws.value });
    queuePrompt(node);
}

function triggerRectRefine(node, rect) {
    const wrp = findWidget(node, "rect_points");
    const ws = findWidget(node, "click_seq");
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    if (!wrp || !ws) {
        dbg("ERROR: rect widgets not found", { wrp: !!wrp, ws: !!ws });
        return;
    }

    // Round to ints; the backend rasterises into latent space anyway.
    const [x1, y1, x2, y2] = rect.map(v => Math.round(v));
    setWidget(wrp, JSON.stringify([[x1, y1, x2, y2]]));
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // Stash a fallback target at the rect centre (some downstream UI
    // reads click_x/y).
    if (wx) setWidget(wx, Math.round((x1 + x2) / 2));
    if (wy) setWidget(wy, Math.round((y1 + y2) / 2));
    if (wr) setWidget(wr, false);
    // Clear stroke_points + seg_polygon so a previous paint stroke or a
    // SAM-detected silhouette can't fall through.
    if (wsp) setWidget(wsp, "");
    const wseg = findWidget(node, "seg_polygon");
    if (wseg) setWidget(wseg, "");
    const wmaskpng = findWidget(node, "seg_mask_png");
    if (wmaskpng) setWidget(wmaskpng, "");

    dbg("queueing workflow (smart inpaint rect)", { x1, y1, x2, y2, click_seq: ws.value });
    queuePrompt(node);
}


// ============================================================
// Reset button (canvas-rendered on the title bar)
// ============================================================

// (Title-bar button rect/draw helpers removed when Reset/Undo moved to
// the DOM toggle bar. Earlier versions had resetButtonRect /
// undoButtonRect / drawTitleButton / hitRect / roundedRect here; see
// git history if we ever want canvas-rendered buttons again.)

function triggerUndo(node) {
    const wu = findWidget(node, "undo_seq");
    if (!wu) return;
    // An open Vary chooser refers to the pre-undo history — close it (the
    // backend drops its candidate stash on undo too, so a pick after this
    // would be a no-op at best and target the wrong entry at worst).
    if (isVaryChooserOpen(node)) hideVaryChooser(node);
    setWidget(wu, ((wu.value || 0) + 1) & 0x7FFFFFFF);
    dbg("queue undo", { undo_seq: wu.value });
    queuePrompt(node);
}

// Redo: re-apply the edit Undo most recently removed. Pure restore (like
// Undo), so it just bumps redo_seq and re-queues — Python pops its redo
// stack back onto history. No seed change, no re-sample.
function triggerRedo(node) {
    const wr = findWidget(node, "redo_seq");
    if (!wr) return;
    if (isVaryChooserOpen(node)) hideVaryChooser(node);   // same as Undo
    setWidget(wr, ((wr.value || 0) + 1) & 0x7FFFFFFF);
    dbg("queue redo", { redo_seq: wr.value });
    queuePrompt(node);
}

// Re-roll: redo the most recent edit with a fresh seed, same mask, same
// pre-edit base. Force a NEW random seed even if Seed Ctrl is "fixed" —
// a re-roll is by definition new dice — then bump reroll_seq so Python
// pops the last attempt and re-runs the (unchanged) mask widgets in its
// place. The mask widgets (click_x/y, stroke_points, rect_points,
// seg_polygon) are deliberately left untouched so the same region is
// reused.
function triggerReroll(node) {
    const wr = findWidget(node, "reroll_seq");
    if (!wr) return;
    const wseed = findWidget(node, "seed");
    if (wseed) {
        setWidget(wseed, Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
        syncSeedInput(node);
    }
    setWidget(wr, ((wr.value || 0) + 1) & 0x7FFFFFFF);
    dbg("queue reroll", { reroll_seq: wr.value, seed: wseed && wseed.value });
    queuePrompt(node);
}

// Vary ×4: like Re-roll, but four candidates at once + a visual pick.
// Bumps vary_seq with a fresh base seed (Python derives the other three
// from it); the run stashes candidates server-side and ships previews,
// which onExecuted hands to showVaryChooser. Nothing commits until
// triggerVaryPick.
function triggerVary(node) {
    const restoreW = findWidget(node, "restore_mode");
    if (restoreW && restoreW.value && !isAnySmartMode(node)) {
        _angeloToast("Turn Restore OFF to use Vary — restores are deterministic");
        return;
    }
    const wv = findWidget(node, "vary_seq");
    if (!wv) return;
    const wseed = findWidget(node, "seed");
    if (wseed) {
        setWidget(wseed, Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
        syncSeedInput(node);
    }
    setWidget(wv, ((wv.value || 0) + 1) & 0x7FFFFFFF);
    _angeloToast("Generating 4 variations…");
    dbg("queue vary", { vary_seq: wv.value });
    queuePrompt(node);
}

function showVaryChooser(node, refs) {
    const ov = node._AngeloVaryOverlay;
    const grid = node._AngeloVaryGrid;
    if (!ov || !grid) return;
    grid.innerHTML = "";
    refs.forEach((ref, i) => {
        const cell = document.createElement("div");
        cell.style.cssText = "position:relative; min-height:0; min-width:0; overflow:hidden; "
            + "border:2px solid #555; border-radius:4px; background:#111; cursor:pointer;";
        const img = document.createElement("img");
        img.src = makeViewUrl(ref);
        img.style.cssText = "width:100%; height:100%; object-fit:contain; display:block;";
        img.draggable = false;
        const tag = document.createElement("span");
        tag.textContent = String(i + 1);
        tag.style.cssText = "position:absolute; left:5px; top:5px; padding:1px 7px; "
            + "background:rgba(0,0,0,0.65); border-radius:3px; color:#ffdc50; "
            + "font:bold 12px Arial,sans-serif; pointer-events:none;";
        cell.addEventListener("mouseenter", () => { cell.style.borderColor = "rgba(255,220,80,0.95)"; });
        cell.addEventListener("mouseleave", () => { cell.style.borderColor = "#555"; });
        cell.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            hideVaryChooser(node);
            triggerVaryPick(node, i);
        });
        cell.appendChild(img);
        cell.appendChild(tag);
        grid.appendChild(cell);
    });
    ov.style.display = "flex";
}

function hideVaryChooser(node) {
    if (node._AngeloVaryOverlay) node._AngeloVaryOverlay.style.display = "none";
}

function isVaryChooserOpen(node) {
    return !!(node._AngeloVaryOverlay && node._AngeloVaryOverlay.style.display !== "none");
}

function triggerVaryPick(node, idx) {
    const wp = findWidget(node, "vary_pick");
    const ws = findWidget(node, "vary_pick_seq");
    if (!wp || !ws) return;
    setWidget(wp, idx);
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    dbg("queue vary pick", { vary_pick: idx, vary_pick_seq: ws.value });
    queuePrompt(node);
}

// Quick Photo Refine: the restoration recipe as one click. Python owns the
// whole pass (whole-canvas, denoise 1.0, internal prompt, reference anchor)
// — the JS only bumps the seq, so no toolbar values get mutated.
function triggerQuickPhotoRefine(node) {
    if (isAnySmartMode(node) || isOutpaintMode(node)) return;  // dimmed there anyway
    if (!node._AngeloImg) {
        _angeloToast("Generate or load an image first");
        return;
    }
    const ws = findWidget(node, "quick_refine_seq");
    if (!ws) return;
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    _angeloToast("✨ Quick Photo Refine — restoring the photo…");
    dbg("queue quick photo refine", { quick_refine_seq: ws.value });
    queuePrompt(node);
}

// ⬆ 2× Pixel: pure lanczos upscale, no AI, committed directly as the new
// session base (deterministic — no review step).
function triggerPixelUpscale(node) {
    if (isAnySmartMode(node) || isOutpaintMode(node)) return;  // dimmed there anyway
    if (!node._AngeloImg) {
        _angeloToast("Generate or load an image first");
        return;
    }
    const ws = findWidget(node, "upscale_seq");
    if (!ws) return;
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    _angeloToast("⬆ 2× Pixel — lanczos upscale, new base (history resets)");
    dbg("queue pixel upscale", { upscale_seq: ws.value });
    queuePrompt(node);
}

// ⬇ Shrink: pick a scale factor in a popup (with a live new-dimensions
// readout), then a pure area-resample downscale — no AI — committed as the
// new session base. triggerShrink does the queue; showShrinkPopup is the UI.
function triggerShrink(node, scale) {
    if (isAnySmartMode(node) || isOutpaintMode(node)) return;
    if (!node._AngeloImg) {
        _angeloToast("Generate or load an image first");
        return;
    }
    const scW = findWidget(node, "shrink_scale");
    const sqW = findWidget(node, "shrink_seq");
    if (!scW || !sqW) return;
    setWidget(scW, scale);
    setWidget(sqW, ((sqW.value || 0) + 1) & 0x7FFFFFFF);
    _angeloToast("⬇ Shrink — downscaling, new base (history resets)");
    dbg("queue shrink", { shrink_scale: scale, shrink_seq: sqW.value });
    queuePrompt(node);
}

function showShrinkPopup(node) {
    if (isAnySmartMode(node) || isOutpaintMode(node)) {
        _angeloToast("Shrink is a Refine-mode action");
        return;
    }
    const img = node._AngeloImg;
    if (!img || !img.naturalWidth) {
        _angeloToast("Generate or load an image first");
        return;
    }
    const inW = img.naturalWidth, inH = img.naturalHeight;
    const snap16 = (v) => Math.max(16, Math.round(v / 16) * 16);

    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed; inset:0; background:rgba(0,0,0,0.6); "
        + "display:flex; align-items:center; justify-content:center; z-index:10000; "
        + "font-family:Arial,sans-serif;";
    const modal = document.createElement("div");
    modal.style.cssText = "background:#2a2a2a; color:#ddd; border:1px solid #555; "
        + "border-radius:8px; padding:16px; width:340px; max-width:90vw; "
        + "display:flex; flex-direction:column; gap:10px;";

    const header = document.createElement("div");
    header.textContent = "Shrink Image";
    header.style.cssText = "font-size:14px; font-weight:bold; color:#aaa;";
    modal.appendChild(header);

    const cur = document.createElement("div");
    cur.textContent = `Current: ${inW} × ${inH}`;
    cur.style.cssText = "font-size:12px; color:#888;";
    modal.appendChild(cur);

    const scaleRow = document.createElement("div");
    scaleRow.style.cssText = "display:flex; align-items:center; gap:8px; font-size:13px;";
    const pctInput = document.createElement("input");
    pctInput.type = "number";
    pctInput.min = "5"; pctInput.max = "95"; pctInput.step = "1"; pctInput.value = "50";
    pctInput.style.cssText = "width:64px; background:#1a1a1a; color:#eee; border:1px solid #555; "
        + "border-radius:3px; padding:3px 6px; font-size:13px;";
    const pctLabel = document.createElement("span");
    pctLabel.textContent = "% of original";
    scaleRow.appendChild(pctInput);
    scaleRow.appendChild(pctLabel);
    modal.appendChild(scaleRow);

    const presetRow = document.createElement("div");
    presetRow.style.cssText = "display:flex; gap:6px;";
    [75, 50, 33, 25].forEach((p) => {
        const b = document.createElement("button");
        b.textContent = p + "%";
        b.style.cssText = "flex:1; background:#3a3a3a; color:#ddd; border:1px solid #555; "
            + "border-radius:4px; padding:4px 0; cursor:pointer; font-size:12px;";
        b.addEventListener("click", () => { pctInput.value = String(p); update(); });
        presetRow.appendChild(b);
    });
    modal.appendChild(presetRow);

    const out = document.createElement("div");
    out.style.cssText = "font-size:13px; color:#ffe9b0; font-weight:bold;";
    modal.appendChild(out);

    const clampPct = () => Math.max(5, Math.min(95, parseFloat(pctInput.value) || 50));
    const update = () => {
        const sc = clampPct() / 100;
        out.textContent = `New: ${snap16(inW * sc)} × ${snap16(inH * sc)}`;
    };
    pctInput.addEventListener("input", update);
    update();

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex; justify-content:flex-end; gap:8px; margin-top:6px;";
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = "background:#444; color:#ddd; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;";
    const okBtn = document.createElement("button");
    okBtn.textContent = "Shrink";
    okBtn.style.cssText = "background:rgba(30,120,80,0.95); color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;";
    footer.appendChild(cancelBtn);
    footer.appendChild(okBtn);
    modal.appendChild(footer);

    backdrop.appendChild(modal);
    _angeloShowModal(node, backdrop);

    const close = () => { if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop); };
    cancelBtn.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
    okBtn.addEventListener("click", () => {
        const sc = clampPct() / 100;
        close();
        triggerShrink(node, sc);
    });
}

// ===== Outpaint — directional canvas extension with review-before-commit =====

function isOutpaintMode(node) {
    const w = findWidget(node, "inpainting_mode");
    return !!w && w.value === "Outpaint";
}

// Fire one extension. Direction from the arrows or the edge-click; amount
// + overlap already live in their widgets via the Outpaint row inputs.
function triggerOutpaint(node, dir) {
    const ws = findWidget(node, "outpaint_seq");
    const wd = findWidget(node, "outpaint_dir");
    if (!ws || !wd) return;
    if (!node._AngeloImg) {
        _angeloToast("Generate or load an image first, then outpaint");
        return;
    }
    setWidget(wd, dir);
    node._AngeloPendingOp = "outpaint";
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // The direction just changed — refresh the combined-prompt preview.
    syncOutpaintPromptPreview(node);
    // Ship the current protect circles (kept client-side so they survive
    // Try-again retries without re-painting).
    const wpr = findWidget(node, "outpaint_protect");
    if (wpr) {
        const prot = (node._AngeloProtect || []).map(
            (c) => [Math.round(c[0]), Math.round(c[1]), Math.round(c[2])]);
        setWidget(wpr, prot.length ? JSON.stringify(prot) : "");
    }
    const amtW = findWidget(node, "outpaint_amount");
    const amt = (amtW && amtW.value) || 256;
    const dirLabel = { left: "left", right: "right", up: "up", down: "down", all: "all sides" }[dir] || dir;
    _angeloToast(`Outpainting ${dirLabel} +${amt}px…`);
    dbg("queue outpaint", { dir, amt, outpaint_seq: ws.value });
    queuePrompt(node);
}

function showOutpaintReview(node, ref) {
    const ov = node._AngeloOutpaintOverlay;
    const img = node._AngeloOutpaintImg;
    if (!ov || !img) return;
    if (node._AngeloOutpaintTitle) {
        node._AngeloOutpaintTitle.textContent = "Outpaint preview — keep it?";
    }
    img.src = makeViewUrl(ref);
    ov.style.display = "flex";
}

function hideOutpaintReview(node) {
    if (node._AngeloOutpaintOverlay) node._AngeloOutpaintOverlay.style.display = "none";
}

function isOutpaintReviewOpen(node) {
    return !!(node._AngeloOutpaintOverlay && node._AngeloOutpaintOverlay.style.display !== "none");
}

function triggerOutpaintAccept(node) {
    const ws = findWidget(node, "outpaint_accept_seq");
    if (!ws) return;
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // The protect circles were drawn on the OLD canvas — stale coords now.
    node._AngeloProtect = [];
    const wpr = findWidget(node, "outpaint_protect");
    if (wpr) setWidget(wpr, "");
    syncOutpaintControls(node);
    // Optimistic swap: the approved pixels are ALREADY loaded in the review
    // overlay's <img> — show them on the canvas NOW instead of making the
    // user stare at the old canvas for the whole commit round-trip. The URL
    // is browser-cached so this is instant, and loadIntoCanvas also updates
    // image_w/h + resets the view for the new dimensions. The commit run's
    // own preview (identical pixels) replaces it seamlessly when it lands.
    const rimg = node._AngeloOutpaintImg;
    if (rimg && rimg.src && rimg.complete && rimg.naturalWidth > 0) {
        loadIntoCanvas(node, rimg.src);
    }
    _angeloToast("New base committed — history reset");
    dbg("queue outpaint accept", { outpaint_accept_seq: ws.value });
    queuePrompt(node);
}

// Same direction + amount, fresh seed. The stale stash is simply
// overwritten by the new run.
function triggerOutpaintRetry(node) {
    const ws = findWidget(node, "outpaint_seq");
    if (!ws) return;
    const wseed = findWidget(node, "seed");
    if (wseed) {
        setWidget(wseed, Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
        syncSeedInput(node);
    }
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    _angeloToast("Trying the extension again with a fresh seed…");
    queuePrompt(node);
}

// Which edge (if any) the cursor is near enough to for the edge-click
// extension. Zones are 18% of each dimension; nearest edge wins so the
// corners resolve cleanly to one direction.
function _outpaintEdgeDir(node, p) {
    const img = node._AngeloImg;
    if (!img || !img.naturalWidth) return null;
    const W = img.naturalWidth, H = img.naturalHeight;
    const zx = W * 0.18, zy = H * 0.18;
    const cands = [];
    if (p.pixelX <= zx) cands.push(["left", p.pixelX / zx]);
    if (W - p.pixelX <= zx) cands.push(["right", (W - p.pixelX) / zx]);
    if (p.pixelY <= zy) cands.push(["up", p.pixelY / zy]);
    if (H - p.pixelY <= zy) cands.push(["down", (H - p.pixelY) / zy]);
    if (!cands.length) return null;
    cands.sort((a, b) => a[1] - b[1]);
    return cands[0][0];
}

// Live preview of the EXACT combined prompt the next outpaint will encode.
// Composition MUST stay in lockstep with Python's outpaint conditioning
// (instruction prepend/append around the trimmed Area text).
function syncOutpaintPromptPreview(node) {
    const wrapEl = node._AngeloOutpaintPromptWrap;
    if (!wrapEl) return;
    const show = isOutpaintMode(node);
    const next = show ? "flex" : "none";
    if (wrapEl.style.display !== next) wrapEl.style.display = next;
    if (!show) return;
    // Mirror the order selector from the widget.
    const posW = findWidget(node, "outpaint_instruction_pos");
    const pos = (posW && posW.value) === "append" ? "append" : "prepend";
    const sel = node._AngeloOutpaintOrderSelect;
    if (sel && sel._AngeloSelect) {
        const want = pos === "append" ? "My text first" : "Instruction first";
        if (sel._AngeloSelect.value !== want) sel._AngeloSelect.value = want;
    }
    const dirW = findWidget(node, "outpaint_dir");
    const dir = (dirW && dirW.value) || "right";
    const instr = _Angelo_OUTPAINT_INSTRUCTIONS[dir] || _Angelo_OUTPAINT_INSTRUCTIONS.right;
    const apW = findWidget(node, "area_prompt");
    const txtW = findWidget(node, "area_text_positive");
    const userTxt = (apW && apW.value) ? String(txtW?.value || "").trim() : "";
    let combined;
    if (userTxt && pos === "append") {
        combined = userTxt.replace(/[ .,]+$/, "") + ". " + instr.trim();
    } else {
        combined = instr + userTxt;
    }
    const pv = node._AngeloOutpaintPromptPreview;
    if (pv && pv.textContent !== combined) pv.textContent = combined;
}

// Outpaint row visibility + input mirrors + protect-chip state.
function syncOutpaintControls(node) {
    const row = node._AngeloOutpaintRow;
    if (row) {
        const next = isOutpaintMode(node) ? "flex" : "none";
        if (row.style.display !== next) row.style.display = next;
    }
    _syncNumberInput(node._AngeloOutpaintAmountInput, findWidget(node, "outpaint_amount")?.value);
    _syncNumberInput(node._AngeloOutpaintOverlapInput, findWidget(node, "outpaint_overlap")?.value);
    const chip = node._AngeloOutpaintClearProtectBtn;
    if (chip) {
        const n = (node._AngeloProtect || []).length;
        chip.style.display = n ? "" : "none";
        chip.textContent = `✕ Protect (${n})`;
    }
    if (!isOutpaintMode(node)) {
        node._AngeloOutpaintHoverDir = null;
        node._AngeloOutpaintPainting = false;
        // Leaving the mode drops the protect region — the canvas may be
        // edited before the user comes back, making the coords stale.
        if (node._AngeloProtect && node._AngeloProtect.length) {
            node._AngeloProtect = [];
            const wpr = findWidget(node, "outpaint_protect");
            if (wpr) setWidget(wpr, "");
        }
        hideOutpaintReview(node);
    }
    syncOutpaintPromptPreview(node);
}

function triggerReset(node) {
    const wr = findWidget(node, "reset");
    const ws = findWidget(node, "click_seq");
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    if (!wr) return;
    wr.value = true;
    if (wx) wx.value = -1;
    if (wy) wy.value = -1;
    if (ws) ws.value = ((ws.value || 0) + 1) & 0x7FFFFFFF;
    node._AngeloImg = null;
    if (node._AngeloCanvas) {
        const ctx = node._AngeloCanvas.getContext("2d");
        ctx.clearRect(0, 0, node._AngeloCanvas.width, node._AngeloCanvas.height);
    }
    if (node._AngeloPlaceholder) node._AngeloPlaceholder.style.display = "flex";
    app.graph.setDirtyCanvas(true, true);
    queuePrompt(node);
    setTimeout(() => {
        if (wr.value === true) {
            wr.value = false;
            app.graph.setDirtyCanvas(true, true);
        }
    }, 1000);
}


// ============================================================
// Helpers
// ============================================================

function findWidget(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find(w => w.name === name);
}

/**
 * Build a momentary action button (Reset, Undo). Click → onClick().
 * `kind` selects a colour theme so different actions are visually
 * distinct in the bar.
 */
function makeActionButton(label, onClick, kind = "neutral") {
    const themes = {
        reset:   { fg: "#ffe0d0", bg: "rgba(70, 50, 50, 0.95)",  border: "rgba(220, 140, 100, 0.9)" },
        undo:    { fg: "#dde7ff", bg: "rgba(50, 60, 70, 0.95)",  border: "rgba(120, 170, 220, 0.9)" },
        redo:    { fg: "#d2f3e2", bg: "rgba(48, 66, 60, 0.95)",  border: "rgba(110, 200, 160, 0.9)" },
        reroll:  { fg: "#ecdcff", bg: "rgba(58, 50, 72, 0.95)",  border: "rgba(170, 130, 220, 0.9)" },
        vary:    { fg: "#d8eeff", bg: "rgba(40, 62, 82, 0.95)",  border: "rgba(120, 190, 235, 0.9)" },
        quickfix:{ fg: "#fff3d0", bg: "rgba(110, 85, 25, 0.95)", border: "rgba(240, 200, 90, 0.9)" },
        neutral: { fg: "#ccc",    bg: "#2a2a2a",                  border: "#555" },
    };
    const th = themes[kind] || themes.neutral;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.style.cursor = "pointer";
    btn.style.padding = "3px 10px";
    btn.style.fontSize = "11px";
    // Pin the line box so emoji glyphs (✨ ⚡) can't inflate a button's
    // height relative to its text-only neighbours.
    btn.style.lineHeight = "15px";
    btn.style.fontWeight = "bold";
    btn.style.border = `1px solid ${th.border}`;
    btn.style.borderRadius = "3px";
    btn.style.background = th.bg;
    btn.style.color = th.fg;
    btn.style.userSelect = "none";
    btn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
    });
    return btn;
}

/** Thin vertical separator for the toggle bar. */
function makeSeparator() {
    const sep = document.createElement("div");
    sep.style.width = "1px";
    sep.style.alignSelf = "stretch";
    sep.style.background = "#444";
    sep.style.margin = "0 4px";
    return sep;
}

/** A horizontal row of controls inside the toolbar wrapper. */
function makeToolbarRow() {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.flexDirection = "row";
    row.style.flexWrap = "wrap";
    row.style.alignItems = "center";
    row.style.gap = "4px";
    row.style.padding = "4px 6px";
    return row;
}


// ============================================================
// Fullscreen overlay
// ============================================================
//
// Pops the whole editing UI (toolbar + area-prompt box + canvas) into a
// fixed, viewport-filling overlay. We reparent node._AngeloContainer INTACT
// rather than cloning anything, so every existing click/paint/zoom/pan
// handler keeps working — they all map through canvas.getBoundingClientRect()
// which reflects the new (much larger) size — and the ResizeObserver on
// canvasWrap auto-refits the image. On exit the container goes back exactly
// where LiteGraph's DOM widget put it (tracked via a comment placeholder).
//
// We also make a best-effort request for TRUE browser fullscreen on the
// overlay so the browser chrome hides too; if the browser refuses (needs a
// user gesture / disallowed), the plain overlay already covers the ComfyUI
// viewport, so the feature still works.

function isAngeloFullscreen(node) {
    return !!node._AngeloFSOverlay;
}

function toggleAngeloFullscreen(node) {
    if (isAngeloFullscreen(node)) exitAngeloFullscreen(node);
    else enterAngeloFullscreen(node);
}

function syncFullscreenButton(node) {
    const btn = node._AngeloFullscreenBtn;
    if (!btn) return;
    const on = isAngeloFullscreen(node);
    btn.textContent = on ? "⛶ Exit" : "⛶ Fullscreen";
    btn.style.background = on ? "rgba(40, 62, 82, 0.95)" : "#2a2a2a";
    btn.style.borderColor = on ? "rgba(120, 190, 235, 0.9)" : "#555";
    btn.style.color = on ? "#d8eeff" : "#ccc";
    btn.title = on
        ? "Exit fullscreen and return the editor to the node (Esc)."
        : "Pop the editor into a fullscreen overlay — a much bigger canvas for precise "
          + "clicks/paint/zoom. The toolbar and all editing work exactly as normal. "
          + "Esc or this button returns it to the node. Requests true browser fullscreen "
          + "too where the browser allows it.";
}

function enterAngeloFullscreen(node) {
    const container = node._AngeloContainer;
    if (!container || node._AngeloFSOverlay) return;

    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed; inset:0; z-index:99990; background:#0d0d0d; "
        + "display:flex; flex-direction:column;";
    // Keep graph-level pointer/wheel handling from firing underneath — clicks
    // inside the toolbar/canvas still reach their own listeners normally.
    overlay.addEventListener("pointerdown", (e) => e.stopPropagation());
    overlay.addEventListener("wheel", (e) => e.stopPropagation(), { passive: false });

    // Remember where the container lives so we can put it back precisely.
    const placeholder = document.createComment("angelo-fullscreen-slot");
    node._AngeloFSPrevParent = container.parentNode;
    node._AngeloFSPlaceholder = placeholder;
    node._AngeloFSPrevCss = container.style.cssText;
    if (container.parentNode) container.parentNode.insertBefore(placeholder, container);

    // Fill the overlay. Same flex-column shape as the in-node container, so
    // the toolbar keeps its height and canvasWrap flex-grows into the rest.
    container.style.cssText = "width:100%; height:100%; flex:1 1 auto; min-height:0; "
        + "border:none; border-radius:0; background:#1a1a1a; overflow:hidden; "
        + "display:flex; flex-direction:column;";
    overlay.appendChild(container);

    // Floating Exit button — the discoverable escape hatch when true browser
    // fullscreen isn't granted (Esc also works; see the keydown handler).
    const exitBtn = document.createElement("button");
    exitBtn.type = "button";
    exitBtn.textContent = "✕ Exit Fullscreen (Esc)";
    exitBtn.title = "Return the editor to the node (Esc).";
    exitBtn.style.cssText = "position:absolute; right:12px; top:10px; z-index:2; cursor:pointer; "
        + "padding:5px 12px; font-size:12px; font-weight:bold; border-radius:4px; "
        + "border:1px solid rgba(255,255,255,0.35); background:rgba(30,30,30,0.92); color:#eee; "
        + "user-select:none;";
    for (const ev of ["pointerdown", "mousedown"]) {
        exitBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    exitBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        exitAngeloFullscreen(node);
    });
    overlay.appendChild(exitBtn);

    document.body.appendChild(overlay);
    node._AngeloFSOverlay = overlay;

    // Esc-to-close — self-contained so it doesn't depend on canvas hover.
    // Defers to any open sub-overlay (detect / vary / outpaint) so the first
    // Esc dismisses that and a second Esc leaves fullscreen.
    const onKey = (e) => {
        if (e.key !== "Escape") return;
        if (isOutpaintReviewOpen(node) || isVaryChooserOpen(node)
            || (node._AngeloDetections && node._AngeloDetections.length)) return;
        // A modal popup (Smart Phrasing / Load Image / Shrink) is open
        // inside the overlay — its own Esc handling closes it; exiting
        // fullscreen here would destroy it (and silently discard e.g. a
        // pasted image). Defer; the next Esc exits fullscreen.
        if (overlay.querySelector(".angelo-modal-backdrop")) return;
        e.preventDefault();
        e.stopPropagation();
        exitAngeloFullscreen(node);
    };
    node._AngeloFSKeyHandler = onKey;
    document.addEventListener("keydown", onKey, true);

    // If the user drops OUT of true browser-fullscreen (e.g. the browser eats
    // Esc to exit fullscreen before our keydown sees it), tear the overlay
    // down to match, so the two never get out of sync.
    const onFsChange = () => {
        if (!document.fullscreenElement && node._AngeloFSOverlay) {
            exitAngeloFullscreen(node);
        }
    };
    node._AngeloFSFsChangeHandler = onFsChange;
    document.addEventListener("fullscreenchange", onFsChange);

    // Best-effort true fullscreen (hides browser chrome). Rejection is fine.
    if (overlay.requestFullscreen) {
        try { overlay.requestFullscreen().catch(() => {}); }
        catch (e) { /* older browsers: ignore */ }
    }

    // Refit the image to the now-huge canvas area. rAF lets the overlay lay
    // out first so clientWidth/Height are real before we measure.
    requestAnimationFrame(() => {
        try { resetView(node); redrawCanvasWithOverlays(node); }
        catch (e) { dbg("fullscreen refit threw", e); }
    });

    syncFullscreenButton(node);
}

function exitAngeloFullscreen(node) {
    const overlay = node._AngeloFSOverlay;
    const container = node._AngeloContainer;
    if (!overlay || !container) return;

    // Detach listeners FIRST so tearing down browser-fullscreen below can't
    // re-enter this via the fullscreenchange handler.
    if (node._AngeloFSKeyHandler) {
        document.removeEventListener("keydown", node._AngeloFSKeyHandler, true);
        node._AngeloFSKeyHandler = null;
    }
    if (node._AngeloFSFsChangeHandler) {
        document.removeEventListener("fullscreenchange", node._AngeloFSFsChangeHandler);
        node._AngeloFSFsChangeHandler = null;
    }
    if (document.fullscreenElement) {
        try { document.exitFullscreen(); } catch (e) { /* noop */ }
    }

    // Put the container back exactly where it came from.
    container.style.cssText = node._AngeloFSPrevCss || "";
    const placeholder = node._AngeloFSPlaceholder;
    const prevParent = node._AngeloFSPrevParent;
    if (placeholder && placeholder.parentNode) {
        placeholder.parentNode.insertBefore(container, placeholder);
        placeholder.parentNode.removeChild(placeholder);
    } else if (prevParent) {
        prevParent.appendChild(container);
    }
    node._AngeloFSPlaceholder = null;
    node._AngeloFSPrevParent = null;
    node._AngeloFSPrevCss = null;

    // Rescue any open modal popup (Smart Phrasing / Load Image / Shrink)
    // before the overlay is destroyed — it was parented into the overlay to
    // render above it, and must not silently die with it (e.g. a pasted
    // image awaiting the Load Image confirm).
    for (const md of Array.from(overlay.querySelectorAll(":scope > .angelo-modal-backdrop"))) {
        document.body.appendChild(md);
    }

    if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    node._AngeloFSOverlay = null;

    requestAnimationFrame(() => {
        try { resetView(node); redrawCanvasWithOverlays(node); }
        catch (e) { dbg("fullscreen restore refit threw", e); }
    });

    syncFullscreenButton(node);
}

/**
 * Build a click-to-toggle button. Returns the DOM element; the caller
 * is responsible for syncing its visual state to whatever underlying
 * widget value it's bound to (call syncPersistentMaskToggle / similar on
 * the relevant node after the widget value changes).
 */
function makeToggleButton(label, onToggle) {
    const btn = document.createElement("button");
    btn.type = "button";
    // State is shown by colour alone (lit fill = ON, flat grey = OFF) —
    // no ": ON/OFF" text suffix. The suffix nearly doubled each button's
    // width and the row holds five toggles; creative tools light buttons
    // up rather than labelling them.
    btn.textContent = label;
    btn.dataset.state = "off";
    btn.style.cursor = "pointer";
    btn.style.padding = "3px 10px";
    btn.style.fontSize = "11px";
    // Same pinned line box as makeActionButton — every button in the
    // row computes the same height.
    btn.style.lineHeight = "15px";
    btn.style.fontWeight = "bold";
    btn.style.border = "1px solid #555";
    btn.style.borderRadius = "3px";
    btn.style.background = "#2a2a2a";
    btn.style.color = "#bbb";
    btn.style.userSelect = "none";
    btn._AngeloLabel = label;
    btn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onToggle();
    });
    return btn;
}

// --- Sync helpers: keep DOM toolbar controls in lockstep with the
//     underlying widget values. Called once on node creation (so persisted
//     widget state reflects in the UI) and after every toggle click.

function _syncToggle(btn, widgetValue, onColor) {
    if (!btn) return;
    const on = !!widgetValue;
    btn.dataset.state = on ? "on" : "off";
    btn.style.background = on ? onColor.bg : "#2a2a2a";
    btn.style.color = on ? "#fff" : "#bbb";
    btn.style.borderColor = on ? onColor.border : "#555";
}

const _TOGGLE_ON_COLORS = {
    blue:   { bg: "rgba(20, 80, 140, 0.95)",  border: "rgba(120, 170, 220, 0.9)" },
    green:  { bg: "rgba(30, 120, 80, 0.95)",  border: "rgba(140, 220, 170, 0.9)" },
    purple: { bg: "rgba(95, 50, 130, 0.95)",  border: "rgba(180, 140, 220, 0.9)" },
    teal:   { bg: "rgba(30, 110, 130, 0.95)", border: "rgba(140, 200, 220, 0.9)" },
    amber:  { bg: "rgba(160, 110, 30, 0.95)", border: "rgba(230, 185, 110, 0.9)" },
    sky:    { bg: "rgba(40, 100, 150, 0.95)", border: "rgba(130, 195, 235, 0.9)" },
    rose:   { bg: "rgba(150, 45, 70, 0.95)",  border: "rgba(230, 120, 150, 0.9)" },
    plum:   { bg: "rgba(110, 45, 120, 0.95)", border: "rgba(195, 130, 220, 0.9)" },
};

function syncPersistentMaskToggle(node) {
    // Backend forces persistent_mask OFF in Smart Guided Inpaint (no
    // mask to persist) — reflect that as OFF rather than the stale
    // widget value.
    const effective = isSmartGuidedInpaintMode(node)
        ? false
        : findWidget(node, "persistent_mask")?.value;
    _syncToggle(node._AngeloPersistentMaskToggle, effective, _TOGGLE_ON_COLORS.blue);
}

function syncAreaPromptToggle(node) {
    // Backend forces area_prompt=True in Smart Inpaint regardless of
    // widget state — reflect that in the displayed toggle so it reads
    // ON instead of misleadingly showing the underlying widget value.
    const effective = isSmartInpaintMode(node)
        ? true
        : findWidget(node, "area_prompt")?.value;
    _syncToggle(node._AngeloAreaPromptToggle, effective, _TOGGLE_ON_COLORS.purple);
}

function syncPaintModeToggle(node) {
    _syncToggle(node._AngeloPaintModeToggle, findWidget(node, "paint_mode")?.value, _TOGGLE_ON_COLORS.teal);
}

function syncRestoreToggle(node) {
    // Backend honours restore_mode in Refine only — the Smart modes ignore
    // it, so display OFF there rather than a stale widget value.
    const effective = isAnySmartMode(node)
        ? false
        : findWidget(node, "restore_mode")?.value;
    _syncToggle(node._AngeloRestoreToggle, effective, _TOGGLE_ON_COLORS.amber);
}

function syncRemoveToggle(node) {
    // Backend honours remove_mode in Refine only (edit models) — the Smart
    // modes ignore it, so display OFF there rather than a stale widget value.
    const effective = isAnySmartMode(node)
        ? false
        : findWidget(node, "remove_mode")?.value;
    _syncToggle(node._AngeloRemoveToggle, effective, _TOGGLE_ON_COLORS.rose);
}

function syncQuickPromptSelect(node) {
    _syncDropdownWrap(node._AngeloQuickPromptSelect, findWidget(node, "quick_prompt_mode")?.value);
}

function syncReferenceControls(node) {
    // Effective OFF in the Smart modes / Outpaint (their own reference
    // logic) regardless of the stored widget value.
    const on = !!(findWidget(node, "refine_reference")?.value)
        && !isAnySmartMode(node) && !isOutpaintMode(node);
    _syncToggle(node._AngeloRefineRefToggle, on, _TOGGLE_ON_COLORS.sky);
    const wrap = node._AngeloRefineRefInput;
    if (wrap) wrap.style.display = on ? "inline-flex" : "none";
    if (node._AngeloRefineRefToggle) {
        node._AngeloRefineRefToggle.style.borderRadius = on ? "3px 0 0 3px" : "3px";
    }
    _syncNumberInput(wrap, findWidget(node, "reference_strength")?.value);
}

function syncFineUpscaleToggle(node) {
    // Backend forces fine_upscaling ON in Smart Inpaint and OFF in
    // Smart Guided Inpaint regardless of widget state — reflect that in
    // the displayed toggle rather than the (stale) underlying value.
    let effective;
    if (isSmartInpaintMode(node)) effective = true;
    else if (isSmartGuidedInpaintMode(node)) effective = false;
    else if (findWidget(node, "remove_mode")?.value) effective = true;  // Remove forces Xtra-Fine on
    else effective = findWidget(node, "fine_upscaling")?.value;
    _syncToggle(node._AngeloFineUpscaleToggle, effective, _TOGGLE_ON_COLORS.green);

    // The Xtra-Fine value group (MP / Max / Ctx Pad / Method + their
    // separator) only applies while Xtra-Fine is effectively ON — show
    // it with the toggle, hide it otherwise. Explicit display values
    // because hiding clears the inline display these wraps were built
    // with (inline-flex for the inputs, default block for the separator).
    const showFine = !!effective;
    const fineGroup = [
        [node._AngeloFineSep, ""],
        [node._AngeloMpInput, "inline-flex"],
        [node._AngeloMaxInput, "inline-flex"],
        [node._AngeloCtxPadInput, "inline-flex"],
        [node._AngeloMethodSelect, "inline-flex"],
    ];
    for (const [el, shown] of fineGroup) {
        if (el) el.style.display = showFine ? shown : "none";
    }
}

// Smart Inpaint mode hard-locks several params on the backend
// regardless of widget state:
//   denoise=1.0, fine_upscaling=ON, fine_context_pad=0, area_prompt=ON
// Dim the corresponding UI controls + paint_mode + click_radius so
// the user can see at a glance that they're not driving anything.
// Feather is NOT locked — a soft edge can help blend the insert, so
// it stays under user control.
const _Angelo_LOCK_SUFFIX = "\n\n[Locked in this Inpaint mode.]";
function _dimControls(node, ids, dim) {
    for (const id of ids) {
        const el = node[id];
        if (!el) continue;
        el.style.opacity = dim ? "0.35" : "";
        el.style.pointerEvents = dim ? "none" : "";
        if (dim) {
            el.title = (el.title?.split("\n\n[")[0] || el.title || "") + _Angelo_LOCK_SUFFIX;
        } else if (el.title && el.title.includes(_Angelo_LOCK_SUFFIX.trim())) {
            el.title = el.title.split(_Angelo_LOCK_SUFFIX)[0];
        }
    }
}

function syncSmartInpaintLockedWidgets(node) {
    const guided = isSmartGuidedInpaintMode(node);
    const anySmart = isSmartInpaintMode(node) || guided;
    const outp = isOutpaintMode(node);

    // Common locks for BOTH smart modes — backend forces these or they
    // don't apply: denoise, fine_upscale toggle, paint_mode, click
    // radius, area_prompt toggle. Outpaint dims the mask-editing set
    // too (the canvas is a direction picker there), but Area Prompt
    // stays LIVE — it describes what fills the new space.
    _dimControls(node, [
        "_AngeloDenoiseInput",
        "_AngeloFineUpscaleToggle",
        "_AngeloPaintModeToggle",
        "_AngeloRestoreToggle",
        "_AngeloRemoveToggle",
        "_AngeloRefGroup",
        "_AngeloCtxPadInput",
    ], anySmart || outp);
    // Click R stays LIVE in Outpaint — it's the protect-brush size there.
    _dimControls(node, ["_AngeloClickRadiusInput"], anySmart);
    _dimControls(node, ["_AngeloAreaPromptToggle"], anySmart);

    // Feather: live in Smart Inpaint (a soft edge can help blend the
    // insert), disabled in Smart Guided (whole-image edit, no mask edge)
    // and in Outpaint (its seam blend is the Overlap input instead).
    _dimControls(node, ["_AngeloFeatherInput"], guided || outp);

    // Persistent Mask: meaningless in Smart Guided (no mask) and in
    // Outpaint (no held mask to re-run). Dimmed + forced OFF there;
    // left alone in Smart Inpaint (re-rolls the rect).
    _dimControls(node, ["_AngeloPersistentMaskToggle"], guided || outp);

    // Re-roll / Vary act on the edit history — confusing mid-outpaint,
    // and the review overlay's "Try again" covers the re-roll need.
    _dimControls(node, ["_AngeloRerollBtn", "_AngeloVaryBtn"], outp);

    // Quick Photo Refine is a Refine-mode action.
    _dimControls(node, ["_AngeloQuickRow"], anySmart || outp);

    // Outpaint row visibility + input mirrors.
    syncOutpaintControls(node);

    // Fine Upscale + Area Prompt toggles' displayed state is forced by
    // the backend (ON for Smart Inpaint; Fine Upscale OFF for Smart
    // Guided; Area Prompt ON for both). Re-run their sync so the labels
    // reflect the forced state, and refresh the Area Prompt box + guided
    // controls visibility.
    syncFineUpscaleToggle(node);
    syncAreaPromptToggle(node);
    syncPersistentMaskToggle(node);
    syncRestoreToggle(node);
    syncRemoveToggle(node);
    // Lite toggle just mirrors quick_lite (no mode forcing).
    _syncToggle(node._AngeloLiteToggle, findWidget(node, "quick_lite")?.value, _TOGGLE_ON_COLORS.teal);
    syncReferenceControls(node);
    syncAreaPromptVisibility(node);
    // Detect row hides in Smart Guided (no mask), shows in Refine/Smart Inpaint.
    syncDetectControls(node);
}

function _syncNumberInput(wrap, widgetValue) {
    if (!wrap || !wrap._AngeloInput || widgetValue == null) return;
    if (wrap._AngeloInput.value !== String(widgetValue)) {
        wrap._AngeloInput.value = String(widgetValue);
    }
}

function syncClickRadiusInput(node) {
    _syncNumberInput(node._AngeloClickRadiusInput, findWidget(node, "click_radius")?.value);
}
function syncFeatherInput(node) {
    _syncNumberInput(node._AngeloFeatherInput, findWidget(node, "feather_radius")?.value);
}
function syncDenoiseInput(node) {
    _syncNumberInput(node._AngeloDenoiseInput, findWidget(node, "denoise")?.value);
}
function syncSeedInput(node) {
    _syncNumberInput(node._AngeloSeedInput, findWidget(node, "seed")?.value);
}
function syncSeedCtrlSelect(node) {
    const wrap = node._AngeloSeedCtrlSelect;
    const w = findWidget(node, "seed_control");
    if (!wrap || !w || !wrap._AngeloSelect) return;
    if (wrap._AngeloSelect.value !== String(w.value)) {
        wrap._AngeloSelect.value = String(w.value);
    }
}

/**
 * Mode-state sync: when mode == "Sampler Mode" grey out the toolbar
 * + canvas (controls inert); when "Edit Mode" un-grey. Also
 * auto-forces sampler_seed_control to "fixed" when switching INTO
 * Edit Mode, so subsequent Queue presses don't regenerate the
 * cached base.
 */
function syncModeState(node) {
    const modeW = findWidget(node, "mode");
    if (!modeW) return;
    const inSampler = String(modeW.value) === "Sampler Mode";

    // Row 3 (mode + shared gen config) is ALWAYS active — both modes use
    // steps/cfg/sampler/scheduler, and the Mode dropdown must always work.
    //
    // Refinement control rows (1+2) grey out in Sampler Mode — they don't
    // apply while generating a base.
    const refineRows = node._AngeloRefineRowsWrap;
    if (refineRows) {
        refineRows.style.opacity = inSampler ? "0.4" : "1";
        refineRows.style.pointerEvents = inSampler ? "none" : "auto";
    }
    // Sampler-seed row (4) greys in Edit Mode — that seed group only
    // drives the base generation.
    const samplerSeedRow = node._AngeloSamplerSeedRow;
    if (samplerSeedRow) {
        samplerSeedRow.style.opacity = inSampler ? "1" : "0.4";
        samplerSeedRow.style.pointerEvents = inSampler ? "auto" : "none";
    }
    // Detect row shows only in Edit Mode's masked sub-modes.
    syncDetectControls(node);
    // Cursor: default in Sampler Mode (clicks do nothing); crosshair
    // in Smart Inpaint (rectangle drag); cell when paint mode is on
    // for Refine; crosshair otherwise.
    if (node._AngeloCanvas) {
        if (inSampler || isSmartGuidedInpaintMode(node) || isOutpaintMode(node)) {
            // Sampler Mode: clicks do nothing. Smart Guided: no canvas
            // interaction (location dropdown + Generate button drive it).
            // Outpaint: default until an edge-hover flips it to pointer.
            node._AngeloCanvas.style.cursor = "default";
        } else if (isSmartInpaintMode(node)) {
            node._AngeloCanvas.style.cursor = "crosshair";
        } else {
            node._AngeloCanvas.style.cursor = isPaintModeOn(node) ? "cell" : "crosshair";
        }
    }
}

function syncModeSwitchToFixed(node, prevMode) {
    // Called from the mode widget's callback. If switching INTO Refinement
    // Mode, force sampler_seed_control to "fixed" AND restore sampler_seed
    // to the seed Python actually used for the cached base (which after
    // after-gen control may have drifted from the current widget value).
    //
    // The pre-Edit control value is snapshotted first so switching BACK to
    // Sampler Mode can undo the force (#29): the control is restored, and
    // one after-gen step is applied immediately — without that, the seed
    // widget still holds the base's seed (lock-on-fixed put it there) and
    // the next Queue silently reproduces the identical base, because
    // after-gen control only fires AFTER a run completes.
    const modeW = findWidget(node, "mode");
    if (!modeW) return;
    const nowMode = String(modeW.value);
    if (nowMode === "Edit Mode" && prevMode === "Sampler Mode") {
        const ctrlW = findWidget(node, "sampler_seed_control");
        node._AngeloPreEditSamplerCtrl = ctrlW ? String(ctrlW.value) : null;
        lockSeedToAtRun(node, "sampler_seed", "sampler_seed_control");
        dbg("syncMode: forced sampler_seed_control → fixed + restored sampler_seed");
    } else if (nowMode === "Sampler Mode" && prevMode === "Edit Mode") {
        const stored = node._AngeloPreEditSamplerCtrl;
        if (stored && stored !== "fixed") {
            const ctrlW = findWidget(node, "sampler_seed_control");
            if (ctrlW) setWidget(ctrlW, stored);
            applyAfterGenControl(node, "sampler_seed", "sampler_seed_control");
            syncSamplerSeedCtrlSelect(node);
            dbg("syncMode: restored sampler_seed_control →", stored, "+ advanced seed");
        }
        node._AngeloPreEditSamplerCtrl = null;
    }
}

/**
 * Apply after-gen control (fixed / increment / decrement / randomize) to
 * a seed widget. ComfyUI's built-in seed widgets auto-attach this dropdown;
 * ours are explicit ENUM widgets so we do the modification ourselves.
 * Runs AFTER the response is processed so seed_at_run capture happens
 * first (lock-on-fixed restores from the pre-modification value).
 */
function applyAfterGenControl(node, seedWidgetName, controlWidgetName) {
    const seedW = findWidget(node, seedWidgetName);
    const ctrlW = findWidget(node, controlWidgetName);
    if (!seedW || !ctrlW) return;
    const ctrl = String(ctrlW.value);
    const maxSeed = 0xFFFFFFFFFFFFFFFF;
    let cur = Number(seedW.value || 0);
    let next = cur;
    switch (ctrl) {
        case "fixed":     return;  // no change
        case "increment": next = (cur + 1) % (maxSeed + 1); break;
        case "decrement": next = cur > 0 ? cur - 1 : maxSeed; break;
        case "randomize":
            // 53 bits is the safe integer range; that's plenty of seed
            // entropy for any sampler.
            next = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
            break;
        default: return;
    }
    setWidget(seedW, next);
    // Mirror to the matching toolbar input.
    if (seedWidgetName === "seed") {
        syncSeedInput(node);
    } else if (seedWidgetName === "sampler_seed") {
        syncSamplerSeedInput(node);
    }
    dbg("after-gen", seedWidgetName, ctrl, cur, "→", next);
}

/**
 * "Lock seed to the value Python actually used at the last run." Used when
 * a control widget flips to "fixed" — restores the seed widget to what was
 * sent to Python (before ComfyUI's after-gen modified it), then forces the
 * control to "fixed". The "_AngeloSamplerSeedAtRun" / "_AngeloRefineSeedAtRun"
 * fields are captured from the Angelo_*_seed_at_run keys in the ui message
 * each onExecuted.
 */
function lockSeedToAtRun(node, seedWidgetName, controlWidgetName) {
    const seedW = findWidget(node, seedWidgetName);
    const ctrlW = findWidget(node, controlWidgetName);
    if (!seedW || !ctrlW) return;
    const stored = seedWidgetName === "sampler_seed"
        ? node._AngeloSamplerSeedAtRun
        : node._AngeloRefineSeedAtRun;
    // Always set control to fixed.
    if (ctrlW.value !== "fixed") setWidget(ctrlW, "fixed");
    // Restore seed value if we have a known-used value to fall back to.
    if (stored != null) {
        if (Number(seedW.value) !== Number(stored)) {
            setWidget(seedW, Number(stored));
        }
    }
    // Mirror to toolbar.
    if (seedWidgetName === "seed") {
        syncSeedInput(node);
        syncSeedCtrlSelect(node);
    } else if (seedWidgetName === "sampler_seed") {
        syncSamplerSeedInput(node);
        syncSamplerSeedCtrlSelect(node);
    }
}
function syncMpInput(node) {
    _syncNumberInput(node._AngeloMpInput, findWidget(node, "min_megapixels")?.value);
}
function syncMaxInput(node) {
    _syncNumberInput(node._AngeloMaxInput, findWidget(node, "max_upscale")?.value);
}

function syncCtxPadInput(node) {
    _syncNumberInput(node._AngeloCtxPadInput, findWidget(node, "fine_context_pad")?.value);
}

function syncMethodSelect(node) {
    const wrap = node._AngeloMethodSelect;
    const w = findWidget(node, "resize_method");
    if (!wrap || !w || !wrap._AngeloSelect) return;
    if (wrap._AngeloSelect.value !== String(w.value)) {
        wrap._AngeloSelect.value = String(w.value);
    }
}

// --- Sync helpers for the row 3/4 sampler + generation controls ---
function _syncDropdownWrap(wrap, widgetValue) {
    if (!wrap || !wrap._AngeloSelect || widgetValue == null) return;
    if (wrap._AngeloSelect.value !== String(widgetValue)) {
        wrap._AngeloSelect.value = String(widgetValue);
    }
}
function syncModeSelect(node) {
    _syncDropdownWrap(node._AngeloModeSelect, findWidget(node, "mode")?.value);
}
function syncStepsInput(node) {
    _syncNumberInput(node._AngeloStepsInput, findWidget(node, "steps")?.value);
}
function syncCfgInput(node) {
    _syncNumberInput(node._AngeloCfgInput, findWidget(node, "cfg")?.value);
}
function syncSamplerSelect(node) {
    _syncDropdownWrap(node._AngeloSamplerSelect, findWidget(node, "sampler_name")?.value);
}
function syncSchedulerSelect(node) {
    _syncDropdownWrap(node._AngeloSchedulerSelect, findWidget(node, "scheduler")?.value);
}
function syncSamplerSeedInput(node) {
    _syncNumberInput(node._AngeloSamplerSeedInput, findWidget(node, "sampler_seed")?.value);
}
function syncSamplerSeedCtrlSelect(node) {
    _syncDropdownWrap(node._AngeloSamplerSeedCtrlSelect, findWidget(node, "sampler_seed_control")?.value);
}
function syncSamplerDenoiseInput(node) {
    _syncNumberInput(node._AngeloSamplerDenoiseInput, findWidget(node, "sampler_denoise")?.value);
}
function syncGuidedLocationSelect(node) {
    _syncDropdownWrap(node._AngeloGuidedLocationSelect, findWidget(node, "guided_location")?.value);
}

function syncInpaintModeSelect(node) {
    const wrap = node._AngeloInpaintModeSelect;
    const w = findWidget(node, "inpainting_mode");
    if (!wrap || !w || !wrap._AngeloSelect) return;
    if (wrap._AngeloSelect.value !== String(w.value)) {
        wrap._AngeloSelect.value = String(w.value);
    }
    // Keep the locked-widget UI in sync — the mode value may have
    // changed via a workflow load, undo, or any path other than the
    // dropdown's own click handler.
    syncSmartInpaintLockedWidgets(node);
}

// Reflect ALL persisted widget state into the DOM toolbar controls.
// Call this both on node creation AND on configure (workflow load) —
// onNodeCreated runs BEFORE ComfyUI restores serialized widget values,
// so a sync there alone leaves toggles like Paint Mode showing the
// default instead of the saved value. onConfigure fires after the
// restore, so a second pass there fixes the mismatch.
function syncAllToolbarControls(node) {
    syncPersistentMaskToggle(node);
    syncAreaPromptToggle(node);
    syncPaintModeToggle(node);
    syncRestoreToggle(node);
    syncRemoveToggle(node);
    syncReferenceControls(node);
    syncQuickPromptSelect(node);
    syncPromptSlotButtons(node);
    syncFineUpscaleToggle(node);
    syncClickRadiusInput(node);
    syncFeatherInput(node);
    syncDenoiseInput(node);
    syncSeedInput(node);
    syncSeedCtrlSelect(node);
    syncMpInput(node);
    syncMaxInput(node);
    syncCtxPadInput(node);
    syncMethodSelect(node);
    syncInpaintModeSelect(node);
    // Row 3/4 sampler + generation controls.
    syncModeSelect(node);
    syncStepsInput(node);
    syncCfgInput(node);
    syncSamplerSelect(node);
    syncSchedulerSelect(node);
    syncSamplerSeedInput(node);
    syncSamplerSeedCtrlSelect(node);
    syncSamplerDenoiseInput(node);
    syncGuidedLocationSelect(node);
    syncLoadImageControls(node);
    syncOutpaintControls(node);
    syncDetectControls(node);
    syncSmartInpaintLockedWidgets(node);
    syncAreaPromptBox(node);
    syncAreaPromptVisibility(node);
    syncModeState(node);
}

function makeDropdown(label, options, onChange) {
    const wrap = document.createElement("div");
    wrap.style.display = "inline-flex";
    wrap.style.alignItems = "center";
    wrap.style.gap = "4px";
    wrap.style.fontSize = "11px";
    wrap.style.color = "#bbb";
    wrap.style.padding = "0 4px";

    const lbl = document.createElement("span");
    lbl.textContent = label;
    wrap.appendChild(lbl);

    const sel = document.createElement("select");
    sel.style.fontSize = "11px";
    sel.style.padding = "2px 4px";
    sel.style.border = "1px solid #555";
    sel.style.borderRadius = "3px";
    sel.style.background = "#1a1a1a";
    sel.style.color = "#ddd";
    for (const opt of options) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        sel.appendChild(o);
    }
    sel.addEventListener("change", () => onChange(sel.value));
    sel.addEventListener("mousedown", (e) => e.stopPropagation());
    sel.addEventListener("pointerdown", (e) => e.stopPropagation());
    wrap.appendChild(sel);
    wrap._AngeloSelect = sel;
    return wrap;
}

/**
 * Small inline numeric input with a label. width is the input width in
 * pixels (label adds ~20px). Calls onChange(numericValue) when committed.
 */
function makeNumberInput(label, opts, onChange) {
    const wrap = document.createElement("div");
    wrap.style.display = "inline-flex";
    wrap.style.alignItems = "center";
    wrap.style.gap = "4px";
    wrap.style.fontSize = "11px";
    wrap.style.color = "#bbb";
    wrap.style.padding = "0 4px";

    const lbl = document.createElement("span");
    lbl.textContent = label;
    wrap.appendChild(lbl);

    const input = document.createElement("input");
    input.type = "number";
    input.min = String(opts.min ?? 0);
    input.max = String(opts.max ?? 100);
    input.step = String(opts.step ?? 1);
    input.style.width = (opts.width || 60) + "px";
    input.style.padding = "2px 4px";
    input.style.fontSize = "11px";
    input.style.border = "1px solid #555";
    input.style.borderRadius = "3px";
    input.style.background = "#1a1a1a";
    input.style.color = "#ddd";
    input.addEventListener("change", () => {
        let v = parseFloat(input.value);
        if (!isFinite(v)) v = opts.min ?? 0;
        const lo = opts.min ?? -Infinity, hi = opts.max ?? Infinity;
        v = Math.max(lo, Math.min(hi, v));
        input.value = String(v);
        onChange(v);
    });
    // Stop click on the input from selecting / dragging the node behind it.
    input.addEventListener("mousedown", (e) => e.stopPropagation());
    input.addEventListener("pointerdown", (e) => e.stopPropagation());
    wrap.appendChild(input);
    wrap._AngeloInput = input;
    return wrap;
}

function setWidget(widget, value) {
    if (!widget) return;
    widget.value = value;
    // Some ComfyUI versions sync the value-for-serialization through the
    // widget's callback rather than reading widget.value directly. Fire
    // it both with and without graph args to maximise compatibility.
    try {
        if (typeof widget.callback === "function") {
            widget.callback(value, app.canvas, widget.node || null);
        }
    } catch (e) {
        dbg("widget callback threw", widget.name, e);
    }
}

function hideMechanicalWidgets(node) {
    // JS-driven widgets that don't need to clutter the user-visible UI.
    // Skipped while Angelo_DEBUG is on so we can watch them update.
    if (Angelo_DEBUG) return;

    const hideNames = [
        // JS-driven plumbing (never user-visible)
        "click_x", "click_y", "click_seq",
        "image_w", "image_h", "undo_seq",
        "stroke_points", "rect_points",
        // Area Prompt text (driven by the DOM textarea below the canvas)
        "area_text_positive", "area_text_negative",
        // Smart Guided Inpaint location (driven by the DOM dropdown)
        "guided_location",
        // Load Image — driven by the Load Image button + popup
        "loaded_image", "loaded_image_seq", "loaded_resize_mode", "loaded_target_mp",
        // Detect (SAM 3 / YOLO) — driven by the Detect row + click-confirm
        "seg_polygon",
        // Detect touch-up brush — base64 raster mask
        "seg_mask_png",
        // Re-roll button — bumps to re-run the last edit with a new seed
        "reroll_seq",
        // Redo button — bumps to restore an edit that Undo removed
        "redo_seq",
        // Restore brush — driven by the Restore toggle on the toolbar
        "restore_mode",
        // Remove brush — driven by the Remove toggle on the toolbar
        "remove_mode",
        // Prompt slots — JSON state for the Area Prompt slot strip
        "area_prompt_slots",
        // Vary ×4 — driven by the Vary button + chooser overlay
        "vary_seq", "vary_pick", "vary_pick_seq",
        // Outpaint — driven by the Outpaint row + edge-click + review overlay
        "outpaint_seq", "outpaint_dir", "outpaint_amount", "outpaint_overlap",
        "outpaint_accept_seq", "outpaint_protect", "outpaint_instruction_pos",
        // Reference — deprecated bool + the live strength value (Ref box)
        "refine_reference", "reference_strength",
        // Quick Photo Refine — driven by the ✨ button
        "quick_refine_seq",
        // ⬆ 2× Pixel — driven by its button
        "upscale_seq",
        // ⬇ Shrink — driven by the button + its popup
        "shrink_seq", "shrink_scale",
        // ✨ prompt selector + Lite toggle — driven by the dropdown/toggle
        "quick_prompt_mode", "quick_lite",
        // Toolbar-driven (visible via the bar above the canvas)
        "persistent_mask", "area_prompt", "paint_mode", "fine_upscaling",
        "click_radius", "feather_radius", "denoise",
        "seed", "seed_control",
        "min_megapixels", "max_upscale", "resize_method", "fine_context_pad",
        "inpainting_mode",
        // Sampler / generation config — now in toolbar rows 3 & 4
        "mode", "sampler_denoise", "sampler_seed", "sampler_seed_control",
        "steps", "cfg", "sampler_name", "scheduler",
        // Deprecated control — always-on, kept declared for serialization
        "auto_decode",
        // Driven by the Reset button on the toolbar — no need to see the widget too
        "reset",
    ];
    if (!node.widgets) return;
    for (const w of node.widgets) {
        // Hide explicit-name matches PLUS any ComfyUI-auto-added
        // "control_after_generate" dropdowns that may have been
        // attached to seed widgets (legacy auto-attach for widgets
        // literally named "seed"). The Python widgets opt out via
        // `"control_after_generate": False`, but some ComfyUI versions
        // ignore that opt-out, so this pattern-match catches the
        // orphaned dropdowns either way.
        const isAutoControl = typeof w.name === "string"
            && /control_after_generate/i.test(w.name);
        if (!hideNames.includes(w.name) && !isAutoControl) continue;
        // The "type = hidden" trick alone isn't enough — some renderers
        // still reserve the widget's natural height. Belt + suspenders:
        // use ComfyUI's own "converted-widget" type (which the frontend
        // explicitly skips), set hidden=true (newer LiteGraph respects
        // it), zero-out computeSize (-4 cancels the 4px inter-widget
        // gap), and null the draw function so nothing paints even if a
        // renderer iterates over hidden widgets.
        w.type = "converted-widget";
        w.hidden = true;
        w.computeSize = () => [0, -4];
        w.draw = () => {};
    }
}

function makeViewUrl(ref) {
    const params = new URLSearchParams();
    params.set("filename", ref.filename);
    if (ref.subfolder) params.set("subfolder", ref.subfolder);
    if (ref.type) params.set("type", ref.type);
    return `/view?${params.toString()}`;
}
