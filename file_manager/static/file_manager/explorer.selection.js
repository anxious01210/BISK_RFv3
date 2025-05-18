function toggleSelection(itemDiv, itemName, isFolder) {
    const key = `${isFolder ? 'folder' : 'file'}:${itemName}`;
    const index = selectedItems.findIndex(i => i.key === key);
    if (index >= 0) {
        selectedItems.splice(index, 1);
        itemDiv.classList.remove("selected");
    } else {
        selectedItems.push({key, name: itemName, isFolder});
        itemDiv.classList.add("selected");
    }
    updateToolbarVisibility();
    updateInfoPanel();
}

// Expose helper to get full relative media paths (used by embedding script)
// window.getSelectedFilePaths = function () {
//     return selectedItems.map(item => {
//         // Reconstruct relative path
//         let currentPath = currentFolderPath || "/";  // Assumes you store this
//         if (!currentPath.endsWith("/")) currentPath += "/";
//         return currentPath + item.name;
//     });
// };

// window.getSelectedFilePaths = function () {
//     return selectedItems.map(item => {
//         let path = currentFolderPath || "/";
//         if (!path.endsWith("/")) path += "/";
//         return path + item.name;
//     });
// };

window.getSelectedFilePaths = function () {
    let currentPath = document.getElementById("breadcrumb")?.dataset?.path || "/";
    if (!currentPath.endsWith("/")) currentPath += "/";
    return selectedItems.map(item => currentPath + item.name);
};