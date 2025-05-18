document.addEventListener("DOMContentLoaded", function () {
    const contextMenu = document.getElementById("contextMenu");

    // ✅ FIXED: Now current folder is dynamically known
    const currentFolderPath = decodeURIComponent(location.pathname.replace('/file-manager/', ''));


    function getSelectedFilePaths() {
        const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
        const folderPath = window.currentFolderPath || "";
        const paths = Array.from(items).map(el => folderPath + el.getAttribute("data-name"));
        console.log("✅ Selected paths:", paths);
        return paths;
    }


    document.addEventListener("contextmenu", function (event) {
        const target = event.target.closest(".file-item, .folder-item");
        if (!target) return;

        event.preventDefault();
        const selected = document.querySelectorAll(".file-item.selected, .folder-item.selected");
        if (!target.classList.contains("selected")) {
            clearSelections();
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
    const selected = getSelectedFilePaths();
    const detSet = document.getElementById("detSet").value;

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

// if you want to keep it here outside the scope, use it as below
// function getSelectedFilePaths(currentFolderPath) {
//     const items = document.querySelectorAll(".file-item.selected, .folder-item.selected");
//     const paths = Array.from(items).map(el => {
//         return currentFolderPath + el.getAttribute("data-name");
//     });
//     console.log("✅ Selected paths:", paths);
//     return paths;
// }
