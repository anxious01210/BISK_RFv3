// explorer.contextmenu.js

// ✅ Global CSRF utility
function getCSRFToken() {
    return document.cookie.split('; ')
        .find(row => row.startsWith('csrftoken'))
        ?.split('=')[1];
}

function getAccurateSelectedPaths() {
    return Array.from(document.querySelectorAll('.file-item.selected, .folder-item.selected'))
        .map(el => el.getAttribute('data-path'))
        .filter(Boolean);
}

document.addEventListener("DOMContentLoaded", function () {
    const contextMenu = document.getElementById("contextMenu");

    document.addEventListener("contextmenu", function (event) {
        const target = event.target.closest(".file-item, .folder-item");
        if (!target) return;

        event.preventDefault();
        const selected = document.querySelectorAll(".file-item.selected, .folder-item.selected");
        if (!target.classList.contains("selected")) {
            document.querySelectorAll(".file-item.selected, .folder-item.selected").forEach(el => el.classList.remove("selected"));
            target.classList.add("selected");
        }

        contextMenu.style.top = `${event.clientY}px`;
        contextMenu.style.left = `${event.clientX}px`;
        contextMenu.style.display = "block";
    });

    document.addEventListener("click", function () {
        contextMenu.style.display = "none";
    });

    // 📌 Add right-click options
    contextMenu.innerHTML += `<li onclick="openEmbeddingModal()">📊 Run Embedding Script</li>`;
    contextMenu.innerHTML += `<li onclick="openSortFacesModal()">🧠 Sort Faces by DetSet</li>`;
});

function openEmbeddingModal(force = false) {
    if (document.getElementById("embeddingModal")) {
        document.getElementById("embeddingModal").remove();
    }

    const modalHtml = `
    <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
        <div class="modal-content">
            <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
            <div style="padding: 2px 6px; border: 1px solid #555; border-radius: 4px; display: flex; align-items: center; gap: 10px; margin: 6px 0;">
                <label for="detSet">Detection Size:</label>
                <select id="detSet">
                    <option value="auto">auto</option>
                    <option value="320,320">320x320</option>
                    <option value="480,480">480x480</option>
                    <option value="640,640">640x640</option>
                    <option value="800,800">800x800</option>
                    <option value="1024,1024">1024x1024</option>
                    <option value="1280,1280">1280x1280</option>
                    <option value="1600,1600">1600x1600</option>
                    <option value="1920,1920">1920x1920</option>
                    <option value="2048,2048">2048x2048</option>
                </select>
                <label style="padding: 0 8px; align-items: center; border: 1px solid #555; border-radius: 4px; background-color: #333;">
                    <input style="padding: 0 2px;" type="checkbox" id="forceCheckbox"> 🛠️ Force
                </label>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:10px;">
                <button onclick="submitEmbeddingScript()">▶ Run</button>
                <button onclick="closeEmbeddingModal()">❌ Cancel</button>
            </div>
        </div>
    </div>`;
    document.body.insertAdjacentHTML("beforeend", modalHtml);
}

function closeEmbeddingModal() {
    const modal = document.getElementById('embeddingModal');
    if (modal) modal.remove();
}

function submitEmbeddingScript() {
    const selected = getAccurateSelectedPaths();
    const detSet = document.getElementById("detSet").value;
    const force = document.getElementById("forceCheckbox").checked;

    console.log("📤 Payload:", {
        paths: selected,
        det_set: detSet,
        force: force
    });

    if (!selected.length) {
        alert("No files or folders selected.");
        return;
    }

    fetch("/file-manager/run-embeddings/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken(),
        },
        body: JSON.stringify({ paths: selected, det_set: detSet, force: force }),
    })
        .then(res => res.json())
        .then(data => {
            alert(data.message || data.error || "Completed.");
        })
        .catch(() => {
            alert("Failed to run script.");
        });

    closeEmbeddingModal();
}

function openSortFacesModal() {
    if (typeof showSortFacesModal === "function") {
        showSortFacesModal(getAccurateSelectedPaths(), getCSRFToken());
    } else {
        alert("Sort Faces modal script is not loaded.");
    }
}










// // explorer.contextmenu.js
// function getAccurateSelectedPaths() {
//     return Array.from(document.querySelectorAll('.file-item.selected, .folder-item.selected'))
//         .map(el => el.getAttribute('data-path'))
//         .filter(Boolean);
// }
//
// document.addEventListener("DOMContentLoaded", function () {
//     const contextMenu = document.getElementById("contextMenu");
//
//     function getCSRFToken() {
//         return document.cookie.split('; ')
//             .find(row => row.startsWith('csrftoken'))
//             ?.split('=')[1];
//     }
//
//     document.addEventListener("contextmenu", function (event) {
//         const target = event.target.closest(".file-item, .folder-item");
//         if (!target) return;
//
//         event.preventDefault();
//         const selected = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         if (!target.classList.contains("selected")) {
//             document.querySelectorAll(".file-item.selected, .folder-item.selected").forEach(el => el.classList.remove("selected"));
//             target.classList.add("selected");
//         }
//
//         contextMenu.style.top = `${event.clientY}px`;
//         contextMenu.style.left = `${event.clientX}px`;
//         contextMenu.style.display = "block";
//     });
//
//     document.addEventListener("click", function () {
//         contextMenu.style.display = "none";
//     });
//
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal()">📊 Run Embedding Script</li>`;
//     contextMenu.innerHTML += `<li onclick="openSortFacesModal()">🧠 Sort Faces by DetSet</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <div style="padding: 2px 6px; border: 1px solid #555; border-radius: 4px; display: flex; align-items: center; gap: 10px; margin: 6px 0;">
//                 <label for="detSet">Detection Size:</label>
//                 <select id="detSet">
//                     <option value="auto">auto</option>
//                     <option value="320,320">320x320</option>
//                     <option value="480,480">480x480</option>
//                     <option value="640,640">640x640</option>
//                     <option value="800,800">800x800</option>
//                     <option value="1024,1024">1024x1024</option>
//                     <option value="1280,1280">1280x1280</option>
//                     <option value="1600,1600">1600x1600</option>
//                     <option value="1920,1920">1920x1920</option>
//                     <option value="2048,2048">2048x2048</option>
//                 </select>
//                 <label style="padding: 0 8px; align-items: center; border: 1px solid #555; border-radius: 4px; background-color: #333;">
//                     <input style="padding: 0 2px;" type="checkbox" id="forceCheckbox"> 🛠️ Force
//                 </label>
//             </div>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript()">▶ Run</button>
//                 <button onclick="closeEmbeddingModal()">❌ Cancel</button>
//             </div>
//         </div>
//     </div>`;
//     document.body.insertAdjacentHTML("beforeend", modalHtml);
// }
//
// function closeEmbeddingModal() {
//     const modal = document.getElementById('embeddingModal');
//     if (modal) modal.remove();
// }
//
// function submitEmbeddingScript() {
//     const selected = getAccurateSelectedPaths();
//     const detSet = document.getElementById("detSet").value;
//     const force = document.getElementById("forceCheckbox").checked;
//
//     console.log("📤 Payload:", {
//         paths: selected,
//         det_set: detSet,
//         force: force
//     });
//
//     if (!selected.length) {
//         alert("No files or folders selected.");
//         return;
//     }
//
//     fetch("/file-manager/run-embeddings/", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json",
//             "X-CSRFToken": getCSRFToken(),
//         },
//         body: JSON.stringify({ paths: selected, det_set: detSet, force: force }),
//     })
//         .then(res => res.json())
//         .then(data => {
//             alert(data.message || data.error || "Completed.");
//         })
//         .catch(() => {
//             alert("Failed to run script.");
//         });
//
//     closeEmbeddingModal();
// }











// // explorer.contextmenu.js
// function getAccurateSelectedPaths() {
//     return Array.from(document.querySelectorAll('.file-item.selected, .folder-item.selected'))
//         .map(el => el.getAttribute('data-path'))
//         .filter(Boolean);
// }
//
// document.addEventListener("DOMContentLoaded", function () {
//     const contextMenu = document.getElementById("contextMenu");
//
//     function getCSRFToken() {
//         return document.cookie.split('; ')
//             .find(row => row.startsWith('csrftoken'))
//             ?.split('=')[1];
//     }
//
//     function getSelectedFilePaths() {
//         const selectedItems = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         const paths = [];
//
//         selectedItems.forEach(el => {
//             const fullPath = el.getAttribute("data-path");
//             if (fullPath) {
//                 paths.push(fullPath);
//             } else {
//                 console.warn("⚠️ Missing data-path on", el);
//             }
//         });
//
//         console.log("✅ Final data-path resolved paths:", paths);
//         return paths;
//     }
//
//     document.addEventListener("contextmenu", function (event) {
//         const target = event.target.closest(".file-item, .folder-item");
//         if (!target) return;
//
//         event.preventDefault();
//         const selected = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         if (!target.classList.contains("selected")) {
//             document.querySelectorAll(".file-item.selected, .folder-item.selected").forEach(el => el.classList.remove("selected"));
//             target.classList.add("selected");
//         }
//
//         contextMenu.style.top = `${event.clientY}px`;
//         contextMenu.style.left = `${event.clientX}px`;
//         contextMenu.style.display = "block";
//     });
//
//     document.addEventListener("click", function () {
//         contextMenu.style.display = "none";
//     });
//
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal()">📊 Run Embedding Script</li>`;
//     contextMenu.innerHTML += `<li onclick="openSortFacesModal()">🧠 Sort Faces by DetSet</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <div style="padding: 2px 6px; border: 1px solid #555; border-radius: 4px; display: flex; align-items: center; gap: 10px; margin: 6px 0;">
//                 <label for="detSet">Detection Size:</label>
//                 <select id="detSet">
//                     <option value="auto">auto</option>
//                     <option value="320,320">320x320</option>
//                     <option value="480,480">480x480</option>
//                     <option value="640,640">640x640</option>
//                     <option value="800,800">800x800</option>
//                     <option value="1024,1024">1024x1024</option>
//                     <option value="1280,1280">1280x1280</option>
//                     <option value="1600,1600">1600x1600</option>
//                     <option value="1920,1920">1920x1920</option>
//                     <option value="2048,2048">2048x2048</option>
//                 </select>
//                 <label style="padding: 0 8px; align-items: center; border: 1px solid #555; border-radius: 4px; background-color: #333;">
//                     <input style="padding: 0 2px;" type="checkbox" id="forceCheckbox"> 🛠️ Force
//                 </label>
//             </div>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript()">▶ Run</button>
//                 <button onclick="closeEmbeddingModal()">❌ Cancel</button>
//             </div>
//         </div>
//     </div>`;
//     document.body.insertAdjacentHTML("beforeend", modalHtml);
// }
//
// function closeEmbeddingModal() {
//     const modal = document.getElementById('embeddingModal');
//     if (modal) modal.remove();
// }
//
// function submitEmbeddingScript() {
//     // const selected = getSelectedFilePaths();
//     const selected = getAccurateSelectedPaths();
//     const detSet = document.getElementById("detSet").value;
//     const force = document.getElementById("forceCheckbox").checked;
//
//     console.log("📤 Payload:", {
//         paths: selected,
//         det_set: detSet,
//         force: force
//     });
//
//     if (!selected.length) {
//         alert("No files or folders selected.");
//         return;
//     }
//
//     fetch("/file-manager/run-embeddings/", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json",
//             "X-CSRFToken": getCSRFToken(),
//         },
//         body: JSON.stringify({paths: selected, det_set: detSet, force: force}),
//     })
//         .then(res => res.json())
//         .then(data => {
//             alert(data.message || data.error || "Completed.");
//         })
//         .catch(() => {
//             alert("Failed to run script.");
//         });
//
//     closeEmbeddingModal();
// }
//