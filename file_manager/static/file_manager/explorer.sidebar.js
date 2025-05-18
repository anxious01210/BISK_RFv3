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