
const currentFolderPath = decodeURIComponent(location.pathname.replace('/file-manager/', ''));

let currentPath = "";
let selectedItems = [];
let cachedFolders = [];
let cachedFiles = [];
let fileLimit = 50;

function loadFolder(path = "") {
    selectedItems = [];
    updateToolbarVisibility();

    fetch(`/file-manager/list-folder/?path=${encodeURIComponent(path)}`)
        .then(res => res.json())
        .then(data => {
            currentPath = data.current_path;
            cachedFolders = data.folders;
            cachedFiles = data.files;

            renderContents(cachedFolders, cachedFiles);
            updateBreadcrumb(currentPath);
            updateInfoPanel();

            // ✅ This line is critical:
            window.currentFolderPath = currentPath;
        })
        .catch(err => console.error("❌ Failed to load folder:", err));
}


function updateBreadcrumb(path) {
    const container = document.getElementById("breadcrumb");
    const parts = path.split('/').filter(p => p);
    let html = `<a href="#" class="breadcrumb-link" data-path="">/media</a>`;
    let accumulated = "";
    for (const part of parts) {
        accumulated += (accumulated ? '/' : '') + part;
        html += ` / <a href="#" class="breadcrumb-link" data-path="${accumulated}">${part}</a>`;
    }
    container.innerHTML = html;

    document.querySelectorAll(".breadcrumb-link").forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            loadFolder(e.target.dataset.path);
        });
    });
}

function renderFolder(data) {
    const container = document.getElementById("folderContents");
    container.innerHTML = "";

    if (data.folders && data.folders.length > 0) {
        data.folders.forEach(folder => {
            const folderDiv = document.createElement("div");
            folderDiv.className = "folder-item";
            folderDiv.setAttribute("data-name", folder.name);

            folderDiv.innerHTML = `
                <div class="thumb-box"><i class="fa fa-folder"></i></div>
                <span class="file-name">${folder.name}</span>
                <small class="file-size">${folder.size || ""}</small>
            `;
            container.appendChild(folderDiv);
        });
    }

    if (data.files && data.files.length > 0) {
        data.files.forEach(file => {
            const fileDiv = document.createElement("div");
            fileDiv.className = "file-item";
            fileDiv.setAttribute("data-name", file.name);

            fileDiv.innerHTML = `
                <div class="thumb-box"><img src="/media/${file.path}" alt="${file.name}" /></div>
                <span class="file-name">${file.name}</span>
                <small class="file-size">${file.size || ""}</small>
            `;
            container.appendChild(fileDiv);
        });
    }
}