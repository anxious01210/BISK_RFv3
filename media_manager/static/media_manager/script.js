document.addEventListener("DOMContentLoaded", () => {
    // Dropzone logic
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.querySelector("input[type='file']");
    const form = document.getElementById("uploadForm");

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");

        const files = e.dataTransfer.files;
        fileInput.files = files;

        // Auto-submit if files dropped
        form.submit();
    });



    // Handle multi-select file actions
    const actionForm = document.getElementById("actionForm");
    const selectedFilesInput = document.getElementById("selectedFilesInput");

    actionForm.addEventListener("submit", function (e) {
        const selected = Array.from(document.querySelectorAll(".file-select:checked"))
            .map(cb => cb.value);
        selectedFilesInput.value = JSON.stringify(selected);
    });


});

// JavaScript Modal Support for move-to
function openMoveModal() {
    const selected = Array.from(document.querySelectorAll(".file-select:checked"))
        .map(cb => cb.value);

    document.getElementById("moveSelectedInput").value = JSON.stringify(selected);
    document.getElementById("destinationInput").value = "";  // Clear previous selection
    document.getElementById("moveModal").style.display = "block";
    loadFolderList();
}

// function openMoveModal() {
//     const selected = Array.from(document.querySelectorAll(".file-select:checked"))
//         .map(cb => cb.value);
//     document.getElementById("moveSelectedInput").value = JSON.stringify(selected);
//     document.getElementById("moveModal").style.display = "block";
// }

function closeMoveModal() {
    document.getElementById("moveModal").style.display = "none";
}


// To Handle Folder Picker

function loadFolderList() {
    console.log("🔄 Fetching folder list...");
    fetch("/media-manager/folder-tree/")
        .then(res => res.json())
        .then(data => {
            console.log("📁 Folder data received:", data);
            const container = document.getElementById("folderList");
            container.innerHTML = "";

            const rootOption = document.createElement("div");
            rootOption.textContent = "/";
            rootOption.className = "folder-option";
            rootOption.dataset.path = "";
            container.appendChild(rootOption);

            data.forEach(item => {
                const div = document.createElement("div");
                div.textContent = item.path;
                div.className = "folder-option";
                div.dataset.path = item.path;
                container.appendChild(div);
            });

            container.querySelectorAll(".folder-option").forEach(option => {
                option.addEventListener("click", () => {
                    container.querySelectorAll(".folder-option").forEach(o => o.classList.remove("selected"));
                    option.classList.add("selected");
                    document.getElementById("destinationInput").value = option.dataset.path;
                    console.log("✅ Selected folder:", option.dataset.path);
                });
            });
        })
        .catch(err => {
            console.error("⚠️ Error loading folder tree:", err);
            document.getElementById("folderList").textContent = "⚠️ Failed to load folders.";
        });
}


// function loadFolderList() {
//     fetch("/media-manager/folder-tree/")
//         .then(res => res.json())
//         .then(data => {
//             const container = document.getElementById("folderList");
//             container.innerHTML = "";
//
//             // Add root folder manually
//             const rootOption = document.createElement("div");
//             rootOption.textContent = "/";
//             rootOption.className = "folder-option";
//             rootOption.dataset.path = "";
//             container.appendChild(rootOption);
//
//             data.forEach(item => {
//                 const div = document.createElement("div");
//                 div.textContent = item.path;
//                 div.className = "folder-option";
//                 div.dataset.path = item.path;
//                 container.appendChild(div);
//             });
//
//             container.querySelectorAll(".folder-option").forEach(option => {
//                 option.addEventListener("click", () => {
//                     // Clear selection
//                     container.querySelectorAll(".folder-option").forEach(o => o.classList.remove("selected"));
//                     option.classList.add("selected");
//
//                     // Set hidden input
//                     document.getElementById("destinationInput").value = option.dataset.path;
//                 });
//             });
//         })
//         .catch(() => {
//             document.getElementById("folderList").textContent = "⚠️ Failed to load folders.";
//         });
// }

function validateMoveForm() {
    const dest = document.getElementById("destinationInput").value;

    // ✅ Accept empty string ("") as valid (it means /media/)
    if (dest === null || dest === undefined) {
        alert("❗ Please select a destination folder from the list.");
        return false;
    }

    // Don’t reject empty string if it’s selected from the folder list
    return true;
}
