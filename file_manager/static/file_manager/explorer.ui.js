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