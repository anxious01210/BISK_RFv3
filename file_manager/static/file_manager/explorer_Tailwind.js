// explorer.js — Tailwind version with fixed checkmarks + selection visibility

document.addEventListener("DOMContentLoaded", () => {
    let currentPath = "";
    let selectedItems = [];
    let fileLimit = 50;
    let cachedFolders = [];
    let cachedFiles = [];

    function loadFolder(path = "") {
        selectedItems = [];
        updateToolbarVisibility();

        fetch(`/file-manager/list-folder/?path=${encodeURIComponent(path)}`)
            .then(res => res.json())
            .then(data => {
                currentPath = data.current_path;
                cachedFolders = data.folders;
                cachedFiles = data.files;
                updateBreadcrumb(currentPath);
                renderContents(cachedFolders, cachedFiles);
                updateInfoPanel();
            })
            .catch(err => console.error("❌ Failed to load folder:", err));
    }

    function formatBytes(bytes) {
        if (!bytes || bytes <= 0) return '0 B';
        const k = 1024, sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function getIconClass(ext) {
        const map = {
            ".pdf": "fa-file-pdf text-red-500",
            ".csv": "fa-file-csv text-green-500",
            ".txt": "fa-file-lines text-gray-300",
            ".pkl": "fa-box text-purple-400",
            ".zip": "fa-file-zipper text-yellow-300",
            ".mp4": "fa-file-video text-pink-400",
            ".mp3": "fa-file-audio text-pink-400",
            ".doc": "fa-file-word text-blue-500",
            ".docx": "fa-file-word text-blue-500",
            ".xlsx": "fa-file-excel text-green-500",
            ".py": "fa-file-code text-yellow-400",
            ".json": "fa-file-code text-gray-300",
            ".jpg": "fa-file-image text-cyan-300",
            ".png": "fa-file-image text-cyan-300",
            ".log": "fa-file-lines text-gray-400",
            ".xcf": "fa-file-image text-fuchsia-300",
            ".tar": "fa-file-archive text-orange-400",
            ".gz": "fa-file-archive text-orange-400",
            ".avi": "fa-film text-rose-300"
        };
        return map[ext.toLowerCase()] || "fa-file text-white";
    }

    function updateBreadcrumb(path) {
        const container = document.getElementById("breadcrumb");
        const parts = path.split('/').filter(p => p);
        let html = `<a href="#" class="text-blue-400 hover:underline breadcrumb-link" data-path="">/media</a>`;
        let accumulated = "";
        for (const part of parts) {
            accumulated += (accumulated ? '/' : '') + part;
            html += ` / <a href="#" class="text-blue-400 hover:underline breadcrumb-link" data-path="${accumulated}">${part}</a>`;
        }
        container.innerHTML = html;
        document.querySelectorAll(".breadcrumb-link").forEach(link => {
            link.addEventListener("click", (e) => {
                e.preventDefault();
                loadFolder(e.target.dataset.path);
            });
        });
    }

    function updateToolbarVisibility() {
        document.getElementById("actionToolbar").classList.toggle("hidden", selectedItems.length === 0);
    }

    function renderContents(folders, files) {
        const container = document.getElementById("folderContents");
        container.innerHTML = "";
        folders.forEach(folder => {
            const div = createItemElement(folder.name, true, null, null, folder.size);
            container.appendChild(div);
        });
        const limit = (fileLimit === "all") ? files.length : parseInt(fileLimit);
        files.slice(0, limit).forEach(file => {
            const div = createItemElement(file.name, false, file.mime, file.ext, file.size);
            container.appendChild(div);
        });
    }

    function createItemElement(name, isFolder, mime = "", ext = "", size = 0) {
        const div = document.createElement("div");
        div.className = `relative p-2 rounded-lg border border-zinc-700 bg-zinc-800 hover:bg-zinc-700 flex flex-col items-center transition select-none file-item`;

        const readableSize = formatBytes(size);
        const byteLabel = (size === null || size === undefined) ? "" : `${size.toLocaleString()} bytes`;

        const thumb = isFolder
            ? `<div class="text-4xl text-yellow-400"><i class="fas fa-folder"></i></div>`
            : (mime && mime.startsWith("image/")
                ? `<img src="/media/${currentPath}/${name}" alt="${name}" class="w-full h-20 object-cover rounded-sm border border-zinc-600 mb-1">`
                : `<div class="text-3xl text-white"><i class="fas ${getIconClass(ext)}"></i></div>`);

        div.innerHTML = `
            <div class="absolute top-1 left-1 w-5 h-5 flex items-center justify-center bg-zinc-700 text-white rounded-md z-10 checkmark hidden">
                <i class="fas fa-check text-xs"></i>
            </div>
            <div class="w-full h-20 flex items-center justify-center overflow-hidden mb-1">
                ${thumb}
            </div>
            <div class="w-full text-xs text-center leading-tight break-words line-clamp-2" title="${name}">${name}</div>
            <small class="text-[11px] text-gray-400 mt-1" title="${byteLabel}">${readableSize}</small>
        `;

        div.onclick = (e) => {
            const key = `${isFolder ? 'folder' : 'file'}:${name}`;
            const index = selectedItems.findIndex(i => i.key === key);
            const check = div.querySelector(".checkmark");
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                if (index >= 0) {
                    selectedItems.splice(index, 1);
                    div.classList.remove("selected");
                    check.classList.add("hidden");
                } else {
                    selectedItems.push({ key, name, isFolder });
                    div.classList.add("selected");
                    check.classList.remove("hidden");
                }
            } else {
                selectedItems = [{ key, name, isFolder }];
                document.querySelectorAll(".file-item").forEach(i => {
                    i.classList.remove("selected");
                    i.querySelector(".checkmark")?.classList.add("hidden");
                });
                div.classList.add("selected");
                check.classList.remove("hidden");
            }
            updateToolbarVisibility();
            updateInfoPanel();
        };

        div.ondblclick = (e) => {
            const fullPath = `/media/${currentPath}/${name}`;
            if (e.ctrlKey || e.metaKey) {
                window.open(window.location.origin + fullPath, "_blank");
            } else if (isFolder) {
                loadFolder((currentPath ? currentPath + '/' : '') + name);
            } else {
                alert(`🖼️ Modal preview would open for: ${name}`);
            }
        };

        return div;
    }

    document.getElementById("refreshBtn").addEventListener("click", () => {
        loadFolder(currentPath);
    });

    document.getElementById("limitSelect").addEventListener("change", (e) => {
        fileLimit = e.target.value;
        loadFolder(currentPath);
    });

    loadFolder("");
});
