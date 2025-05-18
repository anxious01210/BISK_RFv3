function getCSRFToken() {
    const name = 'csrftoken';
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        const trimmed = cookie.trim();
        if (trimmed.startsWith(name + '=')) {
            return decodeURIComponent(trimmed.substring(name.length + 1));
        }
    }
    return '';
}


function formatETA(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
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