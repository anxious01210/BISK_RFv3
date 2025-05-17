
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
            });
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

    function toggleSelection(itemDiv, itemName, isFolder) {
        const key = `${isFolder ? 'folder' : 'file'}:${itemName}`;
        const index = selectedItems.findIndex(i => i.key === key);
        if (index >= 0) {
            selectedItems.splice(index, 1);
            itemDiv.classList.remove("selected");
        } else {
            selectedItems.push({ key, name: itemName, isFolder });
            itemDiv.classList.add("selected");
        }
        updateToolbarVisibility();
        updateInfoPanel();
    }

    function updateToolbarVisibility() {
        const toolbar = document.getElementById("actionToolbar");
        toolbar.style.display = selectedItems.length > 0 ? "flex" : "none";
    }

    function renderContents(folders, files) {
        const container = document.getElementById("folderContents");
        container.innerHTML = "";

        folders.forEach(folder => {
            const div = createItemElement(folder.name, true, "", "", folder.size);
            container.appendChild(div);
        });

        const limit = (fileLimit === "all") ? files.length : parseInt(fileLimit);
        const limitedFiles = files.slice(0, limit);

        limitedFiles.forEach(file => {
            const div = createItemElement(file.name, false, file.mime, file.ext, file.size);
            container.appendChild(div);
        });
    }

    function createItemElement(name, isFolder, mime = "", ext = "", size = null) {
        const div = document.createElement("div");
        div.className = `file-item ${isFolder ? "folder" : "file"}`;

        const readableSize = formatBytes(size);
        const byteLabel = (size === null || size === undefined) ? "" : `${size.toLocaleString()} bytes`;
        const lowerExt = ext?.toLowerCase() || name.slice(name.lastIndexOf('.')).toLowerCase();
        const imageExtensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"];
        const isImageExt = imageExtensions.includes(lowerExt);
        const isImageMime = mime?.startsWith("image/");
        const shouldRenderImage = isImageExt && isImageMime;

        let thumbHTML = "";
        if (isFolder) {
            thumbHTML = `<div class="icon"><i class="fa-solid fa-folder"></i></div>`;
        } else if (shouldRenderImage) {
            thumbHTML = `<img src="/media/${currentPath}/${name}" alt="${name}">`;
        } else {
            thumbHTML = `<div class="icon">${getIconForExtension(lowerExt)}</div>`;
        }

        div.innerHTML = `
            <div class="checkmark"></div>
            <div class="thumb-box">${thumbHTML}</div>
            <span title="${name}">${name}</span>
            <small class="item-size" title="${byteLabel}">${readableSize}</small>
        `;

        div.onclick = (e) => {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                toggleSelection(div, name, isFolder);
            } else {
                selectedItems = [];
                document.querySelectorAll(".file-item").forEach(i => i.classList.remove("selected"));
                selectedItems.push({ key: `${isFolder ? "folder" : "file"}:${name}`, name, isFolder });
                div.classList.add("selected");
                updateToolbarVisibility();
                updateInfoPanel();
            }
        };

        div.ondblclick = (e) => {
            const fullPath = `/media/${currentPath}/${name}`;
            if (e.ctrlKey || e.metaKey) {
                window.open(window.location.origin + fullPath, "_blank");
            } else {
                if (isFolder) {
                    loadFolder((currentPath ? currentPath + '/' : '') + name);
                } else {
                    alert(`🖼️ Modal preview would open for: ${name}`);
                }
            }
        };

        return div;
    }

    function formatBytes(bytes) {
        if (bytes === 0 || bytes === null || bytes === undefined) return "—";
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return parseFloat((bytes / Math.pow(1024, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function getIconForExtension(ext) {
        const icons = {
            ".pdf": "fa-file-pdf text-red-500",
            ".csv": "fa-file-csv text-green-500",
            ".txt": "fa-file-lines text-gray-400",
            ".pkl": "fa-box-archive text-purple-400",
            ".zip": "fa-file-zipper text-yellow-400",
            ".gz": "fa-file-zipper text-yellow-400",
            ".mp4": "fa-file-video text-blue-400",
            ".avi": "fa-file-video text-blue-400",
            ".mp3": "fa-file-audio text-pink-400",
            ".doc": "fa-file-word text-blue-700",
            ".docx": "fa-file-word text-blue-700",
            ".xlsx": "fa-file-excel text-green-700",
            ".xls": "fa-file-excel text-green-700",
            ".py": "fa-file-code text-yellow-300",
            ".json": "fa-file-code text-yellow-300",
            ".jpg": "fa-file-image text-orange-500",
            ".jpeg": "fa-file-image text-orange-500",
            ".png": "fa-file-image text-orange-500",
            ".svg": "fa-file-image text-orange-500",
            ".xcf": "fa-paintbrush text-pink-400",
            ".log": "fa-file-lines text-gray-500"
        };
        const icon = icons[ext.toLowerCase()] || "fa-file text-white";
        return `<i class="fa-solid ${icon}"></i>`;
    }

    async function getRecursiveFolderStats(paths) {
        const response = await fetch("/file-manager/analyze-folders/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths })
        });
        return await response.json();
    }

    async function updateInfoPanel() {
        const info = document.getElementById("infoDetails");
        info.innerHTML = "";

        if (selectedItems.length === 1) {
            const item = selectedItems[0];
            const fullPath = `/media/${currentPath}/${item.name}`;
            info.innerHTML += `
                <h3>ℹ️ Selected Item</h3>
                <p><strong>Name:</strong> ${item.name}</p>
                <p><strong>Type:</strong> ${item.isFolder ? "Folder" : "File"}</p>
                <p><strong>Path:</strong> ${fullPath}</p>
                <hr style="margin: 8px 0; border: 1px solid #444;">
            `;
        }

        if (selectedItems.length > 0) {
            let countFiles = 0;
            let countFolders = 0;
            let imageCount = 0;

            selectedItems.forEach(item => {
                if (item.isFolder) {
                    countFolders += 1;
                } else {
                    countFiles += 1;
                    const ext = item.name.split('.').pop().toLowerCase();
                    if (["jpg", "jpeg", "png", "gif", "bmp", "webp"].includes(ext)) {
                        imageCount += 1;
                    }
                }
            });

            info.innerHTML += `
                <h3>📌 Selected Summary</h3>
                <p><strong>Items:</strong> ${selectedItems.length}</p>
                <p><strong>Top-level Files:</strong> ${countFiles}</p>
                <p><strong>Top-level Folders:</strong> ${countFolders}</p>
                <p><strong>Images (top-level):</strong> ${imageCount}</p>
            `;

            const selectedPaths = selectedItems.filter(i => i.isFolder).map(i => `${currentPath}/${i.name}`);
            if (selectedPaths.length > 0) {
                const stats = await getRecursiveFolderStats(selectedPaths);
                info.innerHTML += `
                    <hr style="margin: 10px 0; border: 1px solid #333;">
                    <h3>📌 Selected Summary (Sub-Levels)</h3>
                    <p><strong>All Files:</strong> ${stats.total_files}</p>
                    <p><strong>All Folders:</strong> ${stats.total_folders}</p>
                    <p><strong>Images:</strong> ${stats.total_images}</p>
                `;
            }

            info.innerHTML += `<hr style="margin: 10px 0; border: 1px solid #333;">`;
        }

        const folderCount = cachedFolders.length;
        const fileCount = cachedFiles.length;
        const extensionMap = {};
        for (const file of cachedFiles) {
            const ext = file.ext?.toLowerCase() || "unknown";
            extensionMap[ext] = (extensionMap[ext] || 0) + 1;
        }

        info.innerHTML += `
            <h3>📁 Current Folder Summary</h3>
            <div class="meta-inline">
                <span><strong>Folders:</strong> ${folderCount}</span>
                <span><strong>Files:</strong> ${fileCount}</span>
            </div>
            <hr class="info-divider">
        `;

        const extHtml = Object.entries(extensionMap).map(
            ([ext, count]) => `<div><strong>${ext}:</strong> ${count}</div>`
        ).join("");

        info.innerHTML += `
            <div class="ext-columns" style="margin-top: 6px;">
                ${extHtml}
            </div>
        `;
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            selectedItems = [];
            updateToolbarVisibility();
            document.querySelectorAll(".file-item").forEach(i => i.classList.remove("selected"));
            updateInfoPanel();
        }
    });

    document.getElementById("refreshBtn").addEventListener("click", () => {
        loadFolder(currentPath);
    });

    document.getElementById("limitSelect").addEventListener("change", (e) => {
        fileLimit = e.target.value;
        loadFolder(currentPath);
    });

    document.getElementById("viewToggle").addEventListener("change", (e) => {
        const view = e.target.value;
        const container = document.getElementById("folderContents");
        if (view === "list") {
            container.classList.remove("grid-view");
            container.classList.add("list-view");
        } else {
            container.classList.remove("list-view");
            container.classList.add("grid-view");
        }
    });

    // Upload file(s)
    const uploadInput = document.getElementById("uploadInput");
    if (uploadInput) {
        uploadInput.addEventListener("change", async function () {
            const files = this.files;
            if (!files.length) return;

            const formData = new FormData();
            for (let file of files) {
                formData.append("files", file);
            }
            formData.append("path", currentPath);

            try {
                const res = await fetch("/file-manager/upload/", {
                    method: "POST",
                    body: formData
                });

                const result = await res.json();
                if (result.status === "ok") {
                    alert(`✅ Uploaded ${result.saved.length} file(s).`);
                    loadFolder(currentPath);
                } else {
                    alert("⚠️ Upload failed.");
                }
            } catch (err) {
                console.error("Upload error", err);
                alert("❌ Upload error.");
            }

            this.value = "";
        });
    }

    // Upload folder
    // let pendingFolderFiles = null;

    let pendingFolderFiles = [];

    document.getElementById("folderUploadInput").addEventListener("change", function (e) {
        console.log("📁 Folder input triggered");

        if (this.files.length > 0) {
            pendingFolderFiles = [...this.files];  // ✅ capture actual file list
            console.log("✅ Files ready for upload:", pendingFolderFiles.length);
            document.getElementById("uploadModeModal").style.display = "flex";
        } else {
            console.warn("⚠️ No files selected from folder");
        }

        this.value = ""; // reset so it triggers again
    });


    // const folderUploadInput = document.getElementById("folderUploadInput");
    // if (folderUploadInput) {
    //     folderUploadInput.addEventListener("change", function () {
    //         pendingFolderFiles = this.files;
    //         document.getElementById("uploadModeModal").style.display = "flex";
    //         this.value = ""; // reset for next use
    //     });
    // }

    // Buttons inside modal
    document.getElementById("uploadFlat").onclick = () => {
        console.log("Flat upload clicked");
        handleFolderUpload("flat");
    }
    document.getElementById("uploadPreserve").onclick = () => {
        console.log("Preserve upload clicked");
        handleFolderUpload("preserve");
    }

    async function handleFolderUpload(mode) {
        const files = pendingFolderFiles;
        document.getElementById("uploadModeModal").style.display = "none";

        // console.log("Selected files:", files);
        // console.log("Sample file:", files[0]);
        // console.log("webkitRelativePath:", files[0]?.webkitRelativePath);

        console.log("⚙️ Folder upload started:", mode);
        console.log("files.length =", files.length);
        console.log("First file:", files[0]);
        console.log("webkitRelativePath =", files[0]?.webkitRelativePath);


        if (!files || !files.length) return;

        const formData = new FormData();
        const relativePaths = [];

        for (let file of files) {
            formData.append("files", file);  // Do NOT override file.name
            relativePaths.push(file.webkitRelativePath);
        }

        formData.append("path", currentPath);
        formData.append("folder_mode", mode);
        formData.append("relative_paths", JSON.stringify(relativePaths));  // send paths explicitly

        try {
            const res = await fetch("/file-manager/upload/", {
                method: "POST",
                body: formData
            });

            const result = await res.json();
            if (result.status === "ok") {
                alert(`✅ Uploaded ${result.saved.length} file(s).`);
                loadFolder(currentPath);
            } else {
                alert("⚠️ Upload failed: " + result.message);
            }
        } catch (err) {
            console.error("Upload error", err);
            alert("❌ Upload error.");
        }

        pendingFolderFiles = null;
    }



    document.getElementById("refreshBtn").addEventListener("click", () => {
        loadFolder(currentPath);
    });

    loadFolder("");
});