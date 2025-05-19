// ✅ FIXED: Global declaration of currentFolderPath

// explorer.contextmenu.js
function getAccurateSelectedPaths() {
    return Array.from(document.querySelectorAll('.file-item.selected, .folder-item.selected'))
        .map(el => el.getAttribute('data-path'))
        .filter(Boolean);
}

document.addEventListener("DOMContentLoaded", function () {
    const contextMenu = document.getElementById("contextMenu");

    function getCSRFToken() {
        return document.cookie.split('; ')
            .find(row => row.startsWith('csrftoken'))
            ?.split('=')[1];
    }

    function getSelectedFilePaths() {
        const selectedItems = document.querySelectorAll(".file-item.selected, .folder-item.selected");
        const paths = [];

        selectedItems.forEach(el => {
            const fullPath = el.getAttribute("data-path");
            if (fullPath) {
                paths.push(fullPath);
            } else {
                console.warn("⚠️ Missing data-path on", el);
            }
        });

        console.log("✅ Final data-path resolved paths:", paths);
        return paths;
    }

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

    contextMenu.innerHTML += `<li onclick="openEmbeddingModal()">📊 Run Embedding Script</li>`;
    contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
});

function openEmbeddingModal(force = false) {
    if (document.getElementById("embeddingModal")) {
        document.getElementById("embeddingModal").remove();
    }

    const modalHtml = `
    <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
        <div class="modal-content" style="min-width:300px; max-width:400px;">
            <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
            <label for="detSet">Detection Size:</label>
            <select id="detSet" style="width:100%; margin:6px 0;">
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
            <div style="display:flex; justify-content:space-between; margin-top:10px;">
                <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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

function submitEmbeddingScript(force = false) {
    // const selected = getSelectedFilePaths();
    const selected = getAccurateSelectedPaths();
    const detSet = document.getElementById("detSet").value;

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
        body: JSON.stringify({paths: selected, det_set: detSet, force: force}),
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
//         // 🛠 Get full current folder relative to /media/
//         let currentMediaPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, '')).replace(/\/$/, '');
//         if (currentMediaPath === '') {
//             currentMediaPath = '';
//         } else if (!currentMediaPath.startsWith('media/')) {
//             currentMediaPath = 'media/' + currentMediaPath;
//         }
//         currentMediaPath = currentMediaPath.replace(/^media\//, '');
//
//         selectedItems.forEach(el => {
//             const name = el.getAttribute("data-name");
//             const fullPath = (currentMediaPath ? currentMediaPath + "/" : "") + name;
//             paths.push(fullPath.replace(/^\/+/, ''));
//         });
//
//         console.log("✅ Final resolved paths:", paths);
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
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
//         // Get full current folder relative to /media/
//         // let currentMediaPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, '')).replace(/\/$/, '');
//         let currentMediaPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, '')).replace(/\/$/, '');
//
//         if (currentMediaPath === '') {
//             currentMediaPath = ''; // top-level
//         } else if (!currentMediaPath.startsWith('media/')) {
//             currentMediaPath = 'media/' + currentMediaPath;
//         }
//         currentMediaPath = currentMediaPath.replace(/^media\//, '');
//
//
//
//         selectedItems.forEach(el => {
//             if (el.classList.contains("file-item")) {
//                 const fileName = el.getAttribute("data-name");
//                 const fullPath = (currentMediaPath ? currentMediaPath + "/" : "") + fileName;
//                 paths.push(fullPath.replace(/^\/+/,''));
//             } else if (el.classList.contains("folder-item")) {
//                 const folderName = el.getAttribute("data-name");
//                 const fullPath = (currentMediaPath ? currentMediaPath + "/" : "") + folderName;
//                 paths.push(fullPath.replace(/^\/+/,''));
//             }
//         });
//         // selectedItems.forEach(el => {
//         //     if (el.classList.contains("file-item")) {
//         //         const img = el.querySelector("img");
//         //         if (img) {
//         //             const parts = img.src.split("/media/");
//         //             if (parts.length > 1) {
//         //                 paths.push(parts[1]);
//         //             }
//         //         }
//         //     } else if (el.classList.contains("folder-item")) {
//         //         const folderName = el.getAttribute("data-name");
//         //         const fullPath = (currentMediaPath ? currentMediaPath + "/" : "") + folderName;
//         //         paths.push(fullPath.replace(/^\/+/,''));
//         //     }
//         // });
//
//         console.log("✅ Final resolved paths:", paths);
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
//     console.log("📤 Payload:", {
//         paths: selected,
//         det_set: detSet,
//         force: force
//     });
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
//         // Get full current folder relative to /media/
//         let currentMediaPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, '')).replace(/\/$/, '');
//
//         selectedItems.forEach(el => {
//             if (el.classList.contains("file-item")) {
//                 const img = el.querySelector("img");
//                 if (img) {
//                     const parts = img.src.split("/media/");
//                     if (parts.length > 1) {
//                         paths.push(parts[1]);
//                     }
//                 }
//             } else if (el.classList.contains("folder-item")) {
//                 const folderName = el.getAttribute("data-name");
//                 const fullPath = (currentMediaPath ? currentMediaPath + "/" : "") + folderName;
//                 paths.push(fullPath.replace(/^\/+/, ''));
//                 // paths.push(fullPath);
//             }
//         });
//
//         console.log("✅ Final resolved paths:", paths);
//         return paths;
//     }
//
//
//
//
//     // function getSelectedFilePaths() {
//     //     const selectedItems = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//     //     const paths = [];
//     //
//     //     selectedItems.forEach(el => {
//     //         if (el.classList.contains("file-item")) {
//     //             const img = el.querySelector("img");
//     //             if (img) {
//     //                 const parts = img.src.split("/media/");
//     //                 if (parts.length > 1) {
//     //                     paths.push(parts[1]);
//     //                 }
//     //             }
//     //         } else if (el.classList.contains("folder-item")) {
//     //             const folderName = el.getAttribute("data-name");
//     //             const currentPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, ''));
//     //             const fullPath = (currentPath ? currentPath + "/" : "") + folderName;
//     //             paths.push(fullPath);
//     //         }
//     //     });
//     //
//     //     console.log("✅ Final resolved paths:", paths);
//     //     return paths;
//     // }
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
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













// document.addEventListener("DOMContentLoaded", function () {
//     const contextMenu = document.getElementById("contextMenu");
//
//     document.addEventListener("contextmenu", function (event) {
//         const target = event.target.closest(".file-item, .folder-item");
//         if (!target) return;
//
//         event.preventDefault();
//         const selected = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         if (!target.classList.contains("selected")) {
//             clearSelections();
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// // function getSelectedFilePaths() {
// //     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
// //     const paths = Array.from(items).map(el => {
// //         const img = el.querySelector("img");
// //         if (!img) return null;
// //         const rawSrc = img.getAttribute("src");
// //         const decoded = decodeURIComponent(rawSrc);
// //         return decoded.replace(/^\/media\//, "");
// //     }).filter(p => p);
// //
// //     console.log("✅ Final resolved paths from <img src>:", paths);
// //     return paths;
// // }
//
//
// // function getSelectedFilePaths() {
// //     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
// //     const folderPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, ''));
// //     const results = [];
// //
// //     items.forEach(el => {
// //         const img = el.querySelector("img");
// //         if (img && img.src.includes("/media/")) {
// //             const relativePath = img.src.split("/media/")[1];
// //             results.push(relativePath);
// //         } else {
// //             // Fallback for folders using data-name
// //             const name = el.getAttribute("data-name");
// //             if (name) {
// //                 const relative = folderPath.endsWith("/") ? folderPath + name : folderPath + "/" + name;
// //                 results.push(relative);
// //             }
// //         }
// //     });
// //
// //     console.log("✅ Final resolved paths:", results);
// //     return results;
// // }
//
//
//
// // function getSelectedFilePaths() {
// //     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
// //     const folderPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, ''));
// //     const results = [];
// //
// //     items.forEach(el => {
// //         const img = el.querySelector("img");
// //
// //         if (img && img.src.includes("/media/")) {
// //             // 📸 It's a file (image)
// //             const relativePath = img.src.split("/media/")[1];
// //             results.push(relativePath);
// //         } else {
// //             // 📁 It's a folder
// //             const name = el.getAttribute("data-name");
// //             if (name) {
// //                 const fullPath = folderPath.endsWith("/") ? folderPath + name : folderPath + "/" + name;
// //                 results.push(fullPath);
// //             }
// //         }
// //     });
// //
// //     console.log("✅ Final resolved paths:", results);
// //     return results;
// // }
//
//
//
// // function getSelectedFilePaths() {
// //     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
// //     const folderPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, ''));
// //     const results = [];
// //
// //     items.forEach(el => {
// //         const img = el.querySelector("img");
// //         const name = el.getAttribute("data-name");
// //
// //         if (img && img.src.includes("/media/")) {
// //             // 🖼 Image file
// //             const relativePath = img.src.split("/media/")[1];
// //             results.push(relativePath);
// //         } else if (name) {
// //             // 📁 Folder
// //             const fullPath = folderPath.endsWith("/") ? folderPath + name : folderPath + "/" + name;
// //             results.push(fullPath);
// //         }
// //     });
// //
// //     console.log("✅ Final resolved paths:", results);
// //     return results;
// // }
//
//
//
//
// function getSelectedFilePaths() {
//     const selectedItems = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//     const paths = [];
//
//     selectedItems.forEach(el => {
//         if (el.classList.contains("file-item")) {
//             const img = el.querySelector("img");
//             if (img) {
//                 const parts = img.src.split("/media/");
//                 if (parts.length > 1) {
//                     paths.push(parts[1]);
//                 }
//             }
//         } else if (el.classList.contains("folder-item")) {
//             const folderName = el.getAttribute("data-name");
//             const currentPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, ''));
//             const fullPath = (currentPath ? currentPath + "/" : "") + folderName;
//             paths.push(fullPath);
//         }
//     });
//
//     console.log("✅ Final resolved paths:", paths);
//     return paths;
// }
//
//
//
//
//
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
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












// document.addEventListener("DOMContentLoaded", function () {
//     const contextMenu = document.getElementById("contextMenu");
//
//
//     function getSelectedFilePaths() {
//         const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         const paths = Array.from(items).map(el => {
//             const img = el.querySelector("img");
//             if (!img) return null;
//             const rawSrc = img.getAttribute("src");
//             console.log(rawSrc)
//             const decoded = decodeURIComponent(rawSrc);  // Convert %20 back to space
//             return decoded.replace(/^\/media\//, "");     // Remove leading /media/
//         }).filter(p => p);  // Remove nulls
//
//         console.log("✅ Final resolved paths from <img src>:", paths);
//         return paths;
//     }
//
//
//
//
//
//     // function getSelectedFilePaths() {
//     //     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//     //     const safePath = window.currentFolderPath.endsWith("/") ? window.currentFolderPath : window.currentFolderPath + "/";
//     //     const paths = Array.from(items).map(el => safePath + el.getAttribute("data-name"));
//     //     console.log("✅ Selected paths:", paths);
//     //     return paths;
//     // }
//
//     document.addEventListener("contextmenu", function (event) {
//         const target = event.target.closest(".file-item, .folder-item");
//         if (!target) return;
//
//         event.preventDefault();
//         const selected = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         if (!target.classList.contains("selected")) {
//             clearSelections();
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
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
//
//





// // ✅ FIXED: Global declaration of currentFolderPath
// const currentFolderPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, ''));
//
// document.addEventListener("DOMContentLoaded", function () {
//     const contextMenu = document.getElementById("contextMenu");
//
//     function getSelectedFilePaths() {
//         const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         const safePath = currentFolderPath.endsWith("/") ? currentFolderPath : currentFolderPath + "/";
//         const paths = Array.from(items).map(el => safePath + el.getAttribute("data-name"));
//         console.log("✅ Selected paths:", paths);
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
//             clearSelections();
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
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











// Working one
// document.addEventListener("DOMContentLoaded", function () {
//     const contextMenu = document.getElementById("contextMenu");
//
//     // ✅ FIXED: Now current folder is dynamically known
//     const currentFolderPath = decodeURIComponent(location.pathname.replace(/^\/file-manager\/?/, ''));
//
//     function getSelectedFilePaths() {
//         const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         const safePath = currentFolderPath.endsWith("/") ? currentFolderPath : currentFolderPath + "/";
//         const paths = Array.from(items).map(el => safePath + el.getAttribute("data-name"));
//         console.log("✅ Selected paths:", paths);
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
//             clearSelections();
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
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




// document.addEventListener("DOMContentLoaded", function () {
//     const contextMenu = document.getElementById("contextMenu");
//
//     // ✅ FIXED: Now current folder is dynamically known
//     const currentFolderPath = decodeURIComponent(location.pathname.replace('/file-manager/', ''));
//
//     function getSelectedFilePaths() {
//         const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         const safePath = currentFolderPath.endsWith("/") ? currentFolderPath : currentFolderPath + "/";
//         const paths = Array.from(items).map(el => safePath + el.getAttribute("data-name"));
//         console.log("✅ Selected paths:", paths);
//         return paths;
//     }
//
//     // function getSelectedFilePaths() {
//     //     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//     //     // const folderPath = window.currentFolderPath || "";
//     //     // const paths = Array.from(items).map(el => folderPath + el.getAttribute("data-name"));
//     //
//     //     const paths = Array.from(items).map(el => {
//     //         const name = el.getAttribute("data-name");
//     //         const safePath = currentFolderPath.endsWith("/") ? currentFolderPath : currentFolderPath + "/";
//     //         return safePath + name;
//     //     });
//     //
//     //
//     //     console.log("✅ Selected paths:", paths);
//     //     return paths;
//     // }
//
//
//     document.addEventListener("contextmenu", function (event) {
//         const target = event.target.closest(".file-item, .folder-item");
//         if (!target) return;
//
//         event.preventDefault();
//         const selected = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//         if (!target.classList.contains("selected")) {
//             clearSelections();
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
//     contextMenu.innerHTML += `<li onclick="openEmbeddingModal(true)">🛠️ Force Regenerate Embeddings</li>`;
// });
//
// function openEmbeddingModal(force = false) {
//     if (document.getElementById("embeddingModal")) {
//         document.getElementById("embeddingModal").remove();
//     }
//
//     const modalHtml = `
//     <div id="embeddingModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="min-width:300px; max-width:400px;">
//             <h3 style="margin-top:0;">📊 Run Embedding Script</h3>
//             <label for="detSet">Detection Size:</label>
//             <select id="detSet" style="width:100%; margin:6px 0;">
//                 <option value="auto">auto</option>
//                 <option value="320,320">320x320</option>
//                 <option value="480,480">480x480</option>
//                 <option value="640,640">640x640</option>
//                 <option value="800,800">800x800</option>
//                 <option value="1024,1024">1024x1024</option>
//                 <option value="1280,1280">1280x1280</option>
//                 <option value="1600,1600">1600x1600</option>
//                 <option value="1920,1920">1920x1920</option>
//                 <option value="2048,2048">2048x2048</option>
//             </select>
//             <div style="display:flex; justify-content:space-between; margin-top:10px;">
//                 <button onclick="submitEmbeddingScript(${force})">▶ Run</button>
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
// function submitEmbeddingScript(force = false) {
//     const selected = getSelectedFilePaths();
//     const detSet = document.getElementById("detSet").value;
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
// // if you want to keep it here outside the scope, use it as below
// // function getSelectedFilePaths(currentFolderPath) {
// //     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
// //     const paths = Array.from(items).map(el => {
// //         return currentFolderPath + el.getAttribute("data-name");
// //     });
// //     console.log("✅ Selected paths:", paths);
// //     return paths;
// // }
