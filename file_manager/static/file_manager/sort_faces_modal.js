// sort_faces_modal.js
function getCSRFToken() {
    return document.cookie.split('; ')
        .find(row => row.startsWith('csrftoken'))
        ?.split('=')[1];
}

function hexToRgbString(hex) {
    hex = hex.replace("#", "");
    if (hex.length === 3) {
        hex = hex.split("").map(c => c + c).join("");
    }
    const bigint = parseInt(hex, 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return `${r},${g},${b}`;
}

function openSortFacesModal() {
    if (document.getElementById("sortFacesModal")) {
        document.getElementById("sortFacesModal").remove();
    }

    const detSizes = [2048, 1600, 1280, 1024, 896, 800, 768, 704, 640];
    const detSetCheckboxes = detSizes.map(size => `
        <label><input type="checkbox" class="det-set-checkbox" value="${size}x${size}" ${[640,800,1024,1600,2048].includes(size) ? "checked" : ""}> ${size}x${size}</label>`).join("<br>");


    const modalHTML = `
    <div id="sortFacesModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
        <div class="modal-content" style="max-height: 90vh; overflow-y: auto; max-width: 1400px; background: #222; padding: 20px; border-radius: 8px; color: #eee;">
            <h3 style="margin-bottom: 20px;">🧠 Sort Faces by DetSet</h3>
    
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <fieldset style="border: 1px solid #666; border-radius: 5px; padding: 10px;">
                    <legend style="padding: 0 10px; font-weight: bold;">Detection Sizes</legend>
                    ${detSetCheckboxes}
                </fieldset>
    
                <fieldset style="border: 1px solid #666; border-radius: 5px; padding: 10px;">
                    <legend style="padding: 0 10px; font-weight: bold;">Preview Settings</legend>
                    <label><input type="checkbox" id="enablePreviewImage" checked> Enable Preview Image</label><br>
                    <label><input type="checkbox" id="enablePreviewOverlay" checked> Enable Preview Overlay</label><br>
                    <label>Crop Margin (%): <input type="number" id="cropMargin" min="0" max="100" value="30" style="border: 1px solid #ccc; background-color: #111; color: #fff; border-radius: 6px; padding: 4px 6px; width: 80px;"></label>
                </fieldset>
    
                <fieldset style="border: 1px solid #666; border-radius: 5px; padding: 10px;">
                    <legend style="padding: 0 10px; font-weight: bold;">Text Styling</legend>
                    <label>Text Color: <input type="color" id="textColor" value="#ffffff" style="border: 1px solid #ccc; background-color: #111; border-radius: 6px;"></label><br>
                    <label>Text Size: <input type="number" id="textSize" step="0.1" value="0.5" style="border: 1px solid #ccc; background-color: #111; color: #fff; border-radius: 6px; padding: 4px 6px; width: 80px;"></label><br>
                    <label><input type="checkbox" id="textBold" checked> Bold Text</label><br>
                    <label>BG Color: <input type="color" id="textBgColor" value="#000000" style="border: 1px solid #ccc; background-color: #111; border-radius: 6px;"></label><br>
                    <label>BG Opacity: <input type="number" id="textBgOpacity" min="0" max="1" step="0.1" value="0.6" style="border: 1px solid #ccc; background-color: #111; color: #fff; border-radius: 6px; padding: 4px 6px; width: 80px;"></label><br>
                    <label><input type="checkbox" id="textBgEnable" checked> Enable BG</label><br>
                    <label><input type="checkbox" id="customFontEnable" checked> Enable Custom Font</label>
                </fieldset>
    
                <fieldset style="border: 1px solid #666; border-radius: 5px; padding: 10px;">
                    <legend style="padding: 0 10px; font-weight: bold;">Script Controls</legend>
                    <label><input type="checkbox" id="enableTerminalLogs" checked> Show terminal output while running</label><br>
                    <label><input type="checkbox" id="outputUnderMedia" checked> Output under /media/ folder</label><br>
                    <label>Max GPU %: <input type="number" id="maxGpuPercent" min="10" max="100" value="95" style="border: 1px solid #ccc; background-color: #111; color: #fff; border-radius: 6px; padding: 4px 6px; width: 80px;"></label><br>
                    <label>Memory Check Interval: <input type="number" id="memCheckInterval" min="1" max="100" value="10" style="border: 1px solid #ccc; background-color: #111; color: #fff; border-radius: 6px; padding: 4px 6px; width: 80px;"></label>
                </fieldset>
            </div>
    
            <div style="margin-top: 20px;">
                <pre id="sortFacesTerminal" style="background:#000; color:#0f0; padding:10px; height:120px; overflow:auto;"></pre>
                <div id="sortFacesStatusText" style="margin-top:5px; color:#aaa;"></div>
            </div>
    
            <div style="margin-top: 15px; display: flex; justify-content: space-between;">
                <button onclick="submitSortFacesScript()">▶ Run</button>
                <button onclick="document.getElementById('sortFacesModal').remove()">❌ Close</button>
            </div>
        </div>
    </div>`;



    document.body.insertAdjacentHTML("beforeend", modalHTML);
}


async function submitSortFacesScript() {
    const paths = [...document.querySelectorAll(".file-item.selected, .folder-item.selected")]
        .map(el => el.getAttribute("data-path")).filter(Boolean);

    if (!paths.length) {
        alert("No files or folders selected.");
        return;
    }

    const det_sets = [...document.querySelectorAll(".det-set-checkbox:checked")].map(cb => cb.value);

    const options = {
        ENABLE_PREVIEW_IMAGE: document.getElementById("enablePreviewImage").checked,
        ENABLE_PREVIEW_OVERLAY: document.getElementById("enablePreviewOverlay").checked,
        PREVIEW_CROP_MARGIN_PERCENT: parseInt(document.getElementById("cropMargin").value),
        ENABLE_TERMINAL_LOGS: document.getElementById("enableTerminalLogs").checked,
        OUTPUT_UNDER_MEDIA_FOLDER: document.getElementById("outputUnderMedia").checked,
        MAX_GPU_MEMORY_PERCENT: parseInt(document.getElementById("maxGpuPercent").value),
        MEMORY_CHECK_INTERVAL: parseInt(document.getElementById("memCheckInterval").value),
        PREVIEW_TEXT_COLOR: hexToRgbString(document.getElementById("textColor").value),
        PREVIEW_TEXT_SIZE: parseFloat(document.getElementById("textSize").value),
        PREVIEW_TEXT_BOLD: document.getElementById("textBold").checked,
        PREVIEW_TEXT_BG_COLOR: hexToRgbString(document.getElementById("textBgColor").value),
        PREVIEW_TEXT_BG_OPACITY: parseFloat(document.getElementById("textBgOpacity").value),
        ENABLE_TEXT_BG: document.getElementById("textBgEnable").checked,
        ENABLE_CUSTOM_FONT: document.getElementById("customFontEnable").checked,
    };

    const terminal = document.getElementById("sortFacesTerminal");
    const statusText = document.getElementById("sortFacesStatusText");

    terminal.innerText = "✅ Script started... \nwaiting for completion...\n";
    statusText.innerText = "⏳ Running...";

    try {
        const res = await fetch("/file-manager/run-sort-faces/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCSRFToken(),
            },
            body: JSON.stringify({ paths, det_sets, options })
        });

        const data = await res.json();
        if (!data.job_id) throw new Error("No job_id returned");

        const job_id = data.job_id;
        // const streamRes = await fetch(`/file-manager/stream-sort-faces/${job_id}/`);
        const streamRes = await fetch(`/file-manager/stream-sort-faces/${job_id}/`, {
            headers: { "X-Requested-With": "XMLHttpRequest" }
        });

        const reader = streamRes.body.getReader();
        const decoder = new TextDecoder("utf-8");

        let buffer = "";
        let doneReading = false;

        while (!doneReading) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();  // keep the last incomplete line

            lines.forEach(line => {
                terminal.innerText += line + "\n";
                terminal.scrollTop = terminal.scrollHeight;

                if (line.includes("✅ Script completed.")) {
                    statusText.innerText = "✅ Done.";
                    doneReading = true;
                }
            });
        }

    } catch (err) {
        console.error("❌ Script error:", err);
        terminal.innerText = "❌ Script failed to start.";
        statusText.innerText = "❌ Failed.";
    }
}




// async function submitSortFacesScript() {
//     const paths = Array.from(document.querySelectorAll(".file-item.selected, .folder-item.selected"))
//         .map(el => el.getAttribute("data-path"))
//         .filter(Boolean);
//
//     if (!paths.length) {
//         alert("No files or folders selected.");
//         return;
//     }
//
//     const det_sets = Array.from(document.querySelectorAll(".det-set-checkbox:checked"))
//         .map(cb => cb.value);
//
//     const options = {
//         ENABLE_PREVIEW_IMAGE: document.getElementById("enablePreviewImage").checked,
//         ENABLE_PREVIEW_OVERLAY: document.getElementById("enablePreviewOverlay").checked,
//         PREVIEW_CROP_MARGIN_PERCENT: parseInt(document.getElementById("cropMargin").value),
//         ENABLE_TERMINAL_LOGS: document.getElementById("enableTerminalLogs").checked,
//         OUTPUT_UNDER_MEDIA_FOLDER: document.getElementById("outputUnderMedia").checked,
//         MAX_GPU_MEMORY_PERCENT: parseInt(document.getElementById("maxGpuPercent").value),
//         MEMORY_CHECK_INTERVAL: parseInt(document.getElementById("memCheckInterval").value),
//
//         PREVIEW_TEXT_COLOR: hexToRgbString(document.getElementById("textColor").value),
//         PREVIEW_TEXT_SIZE: parseFloat(document.getElementById("textSize").value),
//         PREVIEW_TEXT_BOLD: document.getElementById("textBold").checked,
//
//         PREVIEW_TEXT_BG_COLOR: hexToRgbString(document.getElementById("textBgColor").value),
//         PREVIEW_TEXT_BG_OPACITY: parseFloat(document.getElementById("textBgOpacity").value),
//         ENABLE_TEXT_BG: document.getElementById("textBgEnable").checked,
//         ENABLE_CUSTOM_FONT: document.getElementById("customFontEnable").checked,
//     };
//
//     const terminal = document.getElementById("sortFacesTerminal");
//     const statusText = document.getElementById("sortFacesStatusText");
//
//     console.log("📤 Submitting Sort Faces job:", { paths, det_sets, options });
//     console.log("🧪 Final Options:", options);
//
//     try {
//         const res = await fetch("/file-manager/run-sort-faces/", {
//             method: "POST",
//             headers: {
//                 "Content-Type": "application/json",
//                 "X-CSRFToken": getCSRFToken(),
//             },
//             body: JSON.stringify({ paths, det_sets, options })
//         });
//
//         const data = await res.json();
//         if (!data.job_id) {
//             terminal.innerText = "❌ Script returned no job ID.";
//             statusText.innerText = "❌ Failed to start.";
//             return;
//         }
//
//         const job_id = data.job_id;
//         console.log("✅ job_id received:", job_id);
//         terminal.innerText = "✅ Script started... waiting for completion";
//         statusText.innerText = "⏳ Running...";
//
//         const logRes = await fetch(`/file-manager/stream-sort-faces/${job_id}/`);
//         const reader = logRes.body.getReader();
//         const decoder = new TextDecoder("utf-8");
//
//         let finalLog = "";
//         while (true) {
//             const { value, done } = await reader.read();
//             if (done) break;
//             if (value) {
//                 const chunk = decoder.decode(value, { stream: true });
//                 finalLog += chunk;
//                 terminal.textContent += chunk;
//                 terminal.scrollTop = terminal.scrollHeight;
//             }
//         }
//
//         // Show these only after stream ends
//         terminal.textContent += "\n✅ Script completed.";
//         statusText.innerText = "✅ Done.";
//         terminal.scrollTop = terminal.scrollHeight;
//
//     } catch (err) {
//         console.error("❌ Script failed:", err);
//         terminal.innerText = "❌ Failed to start script.";
//         statusText.innerText = "❌ Failed to start.";
//     }
// }




// function submitSortFacesScript() {
//     const paths = Array.from(document.querySelectorAll(".file-item.selected, .folder-item.selected"))
//         .map(el => el.getAttribute("data-path"))
//         .filter(Boolean);
//
//     if (!paths.length) {
//         alert("No files or folders selected.");
//         return;
//     }
//
//     const det_sets = Array.from(document.querySelectorAll(".det-set-checkbox:checked"))
//         .map(cb => cb.value);
//
//     const options = {
//         ENABLE_PREVIEW_IMAGE: document.getElementById("enablePreviewImage").checked,
//         ENABLE_PREVIEW_OVERLAY: document.getElementById("enablePreviewOverlay").checked,
//         PREVIEW_CROP_MARGIN_PERCENT: parseInt(document.getElementById("cropMargin").value),
//         ENABLE_TERMINAL_LOGS: document.getElementById("enableTerminalLogs").checked,
//         OUTPUT_UNDER_MEDIA_FOLDER: document.getElementById("outputUnderMedia").checked,
//         MAX_GPU_MEMORY_PERCENT: parseInt(document.getElementById("maxGpuPercent").value),
//         MEMORY_CHECK_INTERVAL: parseInt(document.getElementById("memCheckInterval").value),
//
//         PREVIEW_TEXT_COLOR: hexToRgbString(document.getElementById("textColor").value),
//         PREVIEW_TEXT_SIZE: parseFloat(document.getElementById("textSize").value),
//         PREVIEW_TEXT_BOLD: document.getElementById("textBold").checked,
//
//         PREVIEW_TEXT_BG_COLOR: hexToRgbString(document.getElementById("textBgColor").value),
//         PREVIEW_TEXT_BG_OPACITY: parseFloat(document.getElementById("textBgOpacity").value),
//         ENABLE_TEXT_BG: document.getElementById("textBgEnable").checked,
//         ENABLE_CUSTOM_FONT: document.getElementById("customFontEnable").checked,
//     };
//
//     const terminal = document.getElementById("sortFacesTerminal");
//     const statusText = document.getElementById("sortFacesStatusText");
//
//     console.log("📤 Submitting Sort Faces job:", { paths, det_sets, options });
//     console.log("🧪 Final Options:", options);
//
//     fetch("/file-manager/run-sort-faces/", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json",
//             "X-CSRFToken": getCSRFToken(),
//         },
//         body: JSON.stringify({ paths, det_sets, options })
//     })
//     .then(res => res.json())
//     .then(data => {
//         if (data.job_id) {
//             console.log("✅ job_id received:", data.job_id);
//             terminal.innerText = "✅ Script started... waiting for completion";
//             statusText.innerText = "⏳ Running...";
//
//             setTimeout(() => {
//                 terminal.innerText += "\n✅ Script completed. (no live logs)";
//                 statusText.innerText = "✅ Done.";
//             }, 10000);
//         } else {
//             terminal.innerText = "❌ Script returned no job ID.";
//             statusText.innerText = "❌ Failed to start.";
//         }
//     })
//     .catch(err => {
//         console.error("❌ Script failed:", err);
//         terminal.innerText = "❌ Failed to start script.";
//         statusText.innerText = "❌ Failed to start.";
//     });
// }

