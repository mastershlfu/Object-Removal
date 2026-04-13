/* =====================================================
   Globals & Init
===================================================== */
const canvasROI = new fabric.Canvas('canvasROI', { selection: false, backgroundColor: '#000' });
const canvasDetect = new fabric.Canvas('canvasDetect', { selection: true, backgroundColor: '#000' });

let currentFilename = "";
let imgOriginalWidth = 0; let imgOriginalHeight = 0;
let drawMode = false; let isDrawing = false; let rectROI = null; let startX, startY;
let detectedObjects = []; 
let viewScale = 1.0; 

fabric.Object.prototype.set({ cornerColor: '#00e5ff', cornerStyle: 'circle', transparentCorners: false });

/* =====================================================
   1. Upload & Setup Scale
===================================================== */
document.getElementById('upload').onchange = function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    currentFilename = file.name;
    const reader = new FileReader();
    
    reader.onload = (f) => {
        const imgObj = new Image();
        imgObj.src = f.target.result;
        
        imgObj.onload = () => {
            imgOriginalWidth = imgObj.width;
            imgOriginalHeight = imgObj.height;

            setupSingleCanvas(canvasROI, imgObj);
            setupSingleCanvas(canvasDetect, imgObj);

            // Gán ảnh gốc vào Slot 4
            document.getElementById('resultOriginal').src = f.target.result;
            // Clear các ảnh cũ
            document.getElementById('resultMask').removeAttribute('src');
            document.getElementById('resultLama').removeAttribute('src');
            document.getElementById('resultSDXL').removeAttribute('src');

            document.getElementById('classFilterSection').classList.add('hidden');
            document.getElementById('removeBtn').disabled = true;

            setTimeout(fitToScreen, 50);
        }
    };
    reader.readAsDataURL(file);
    this.value = ''; 
};

function setupSingleCanvas(fabricCanvas, imgElement) {
    fabricCanvas.setDimensions({ width: imgOriginalWidth, height: imgOriginalHeight });
    fabricCanvas.clear();
    const fImg = new fabric.Image(imgElement, { selectable: false, evented: false, originX: 'left', originY: 'top' });
    fabricCanvas.setBackgroundImage(fImg, fabricCanvas.renderAll.bind(fabricCanvas));
}

function fitToScreen() {
    if (imgOriginalWidth === 0) return;
    const items = [
        { id: 'wrap1', el: canvasROI.getElement().parentElement },
        { id: 'wrap2', el: canvasDetect.getElement().parentElement },
        { id: 'wrap3', el: document.getElementById('resultMask') },
        { id: 'wrap4', el: document.getElementById('resultOriginal') },
        { id: 'wrap5', el: document.getElementById('resultLama') },
        { id: 'wrap6', el: document.getElementById('resultSDXL') }
    ];

    items.forEach(item => {
        const wrapper = document.getElementById(item.id);
        if(!wrapper || !item.el) return;
        const scale = Math.min((wrapper.clientWidth - 40) / imgOriginalWidth, (wrapper.clientHeight - 40) / imgOriginalHeight);
        if(item.id === 'wrap1') viewScale = scale; 
        
        item.el.style.width = `${imgOriginalWidth}px`;
        item.el.style.height = `${imgOriginalHeight}px`;
        item.el.style.transform = `translate(-50%, -50%) scale(${scale})`;
    });
}
window.addEventListener('resize', () => requestAnimationFrame(fitToScreen));

/* =====================================================
   2. ROI Drawing (Giữ nguyên)
===================================================== */
const drawBtn = document.getElementById('drawBtn');
drawBtn.onclick = () => {
    drawMode = !drawMode;
    drawBtn.innerText = drawMode ? '✏️ Mode: DRAW' : '👁️ Mode: VIEW';
    drawBtn.className = drawMode ? 'btn btn-warning' : 'btn btn-primary';
    canvasROI.discardActiveObject(); canvasROI.requestRenderAll();
    canvasROI.defaultCursor = drawMode ? 'crosshair' : 'default';
};

canvasROI.on('mouse:down', (opt) => {
    if (!drawMode || opt.target) return;
    isDrawing = true;
    const p = canvasROI.getPointer(opt.e);
    startX = p.x; startY = p.y;
    rectROI = new fabric.Rect({ left: startX, top: startY, width: 0, height: 0, fill: 'rgba(255,255,255,0.1)', stroke: '#00e5ff', strokeWidth: 3 / viewScale, selectable: true, hasControls: true });
    canvasROI.add(rectROI); canvasROI.setActiveObject(rectROI);
});
canvasROI.on('mouse:move', (opt) => {
    if (!isDrawing) return;
    const p = canvasROI.getPointer(opt.e);
    rectROI.set({ width: Math.abs(p.x - startX), height: Math.abs(p.y - startY), left: Math.min(p.x, startX), top: Math.min(p.y, startY) });
    canvasROI.requestRenderAll();
});
canvasROI.on('mouse:up', () => { isDrawing = false; if(rectROI) rectROI.setCoords(); });
document.addEventListener('keydown', (e) => { if ((e.key === 'Delete' || e.key === 'Backspace') && canvasROI.getActiveObject()) canvasROI.remove(canvasROI.getActiveObject()); });

/* =====================================================
   3. Scan & Display Logic (CẬP NHẬT 2 MODEL)
===================================================== */
document.getElementById('scanBtn').onclick = async () => {
    const rois = canvasROI.getObjects('rect');
    const payload = {
        image_name: currentFilename,
        boxes: rois.map(r => ({ xmin: Math.round(r.left), ymin: Math.round(r.top), xmax: Math.round(r.left + r.width * r.scaleX), ymax: Math.round(r.top + r.height * r.scaleY) }))
    };

    try {
        const res = await fetch('http://127.0.0.1:8001/submit_boxes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await res.json();
        
        if (data.status === 'success') {
            detectedObjects = []; // Reset mảng
            // Gộp 2 mảng lại nhưng dán nhãn nguồn gốc Model
            if (data.objects.general_objects) {
                data.objects.general_objects.forEach(obj => detectedObjects.push({...obj, sourceModel: 'general'}));
            }
            if (data.objects.custom_objects) {
                data.objects.custom_objects.forEach(obj => detectedObjects.push({...obj, sourceModel: 'custom'}));
            }
            renderDetectedObjects(detectedObjects);
            document.getElementById('removeBtn').disabled = false;
        }
    } catch (err) { console.error(err); alert("Lỗi API Scan"); }
};

function renderDetectedObjects(objects) {
    canvasDetect.clear();
    canvasROI.backgroundImage.clone(bg => canvasDetect.setBackgroundImage(bg, canvasDetect.renderAll.bind(canvasDetect)));
    const classes = {};

    objects.forEach((obj, index) => {
        // Màu mặc định khi KHÔNG chọn (General = Blue, Custom = Green)
        const unselectedColor = obj.sourceModel === 'general' ? '#3b82f6' : '#10b981';
        
        // Trạng thái ban đầu: SELECTED (Red)
        const box = new fabric.Rect({
            left: obj.box[0], top: obj.box[1],
            width: obj.box[2] - obj.box[0], height: obj.box[3] - obj.box[1],
            fill: 'rgba(239, 68, 68, 0.4)', stroke: '#ef4444', strokeWidth: 3 / viewScale,
            objectCaching: false,
            isSelectedForRemoval: true, 
            classLabel: obj.label,
            sourceModel: obj.sourceModel, // Lưu lại model gốc
            unselectedColor: unselectedColor // Lưu mã màu tĩnh
        });

        const text = new fabric.Text(`${obj.label} (${obj.sourceModel})`, {
            left: obj.box[0], top: obj.box[1] - (20/viewScale),
            fontSize: 14 / viewScale, fill: '#ef4444', backgroundColor: 'rgba(0,0,0,0.8)', fontFamily: 'Inter'
        });

        const group = new fabric.Group([box, text], { selectable: true, hasControls: false, lockMovementX: true, lockMovementY: true, dataIndex: index, className: obj.label, hoverCursor: 'pointer' });
        
        group.on('mousedown', () => toggleSelectionState(group));
        canvasDetect.add(group);

        if (!classes[obj.label]) classes[obj.label] = 0;
        classes[obj.label]++;
    });

    renderClassFilter(classes);
    canvasDetect.requestRenderAll();
}

function toggleSelectionState(groupObj, forceState = null) {
    const box = groupObj.getObjects()[0];
    const text = groupObj.getObjects()[1];
    const newState = forceState !== null ? forceState : !box.isSelectedForRemoval;
    box.isSelectedForRemoval = newState;

    if (newState) {
        // SELECTED -> ĐỎ
        box.set({ stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.4)', strokeDashArray: null, strokeWidth: 3 / viewScale });
        text.set({ fill: '#ef4444' });
    } else {
        // UNSELECTED -> VỀ MÀU CỦA MODEL (Xanh dương / Xanh lá)
        box.set({ stroke: box.unselectedColor, fill: `${box.unselectedColor}1A`, strokeDashArray: [5, 5], strokeWidth: 2 / viewScale });
        text.set({ fill: box.unselectedColor });
    }
    canvasDetect.requestRenderAll();
}

function renderClassFilter(classCounts) {
    const container = document.getElementById('classList');
    container.innerHTML = '';
    document.getElementById('classFilterSection').classList.remove('hidden');

    Object.keys(classCounts).forEach(cls => {
        const div = document.createElement('div');
        div.className = 'class-item';
        div.innerHTML = `<input type="checkbox" id="chk-${cls}" checked /><label for="chk-${cls}">${cls}</label><span class="badge">${classCounts[cls]}</span>`;
        div.querySelector('input').onchange = (e) => {
            canvasDetect.getObjects('group').forEach(g => { if (g.className === cls) toggleSelectionState(g, e.target.checked); });
        };
        container.appendChild(div);
    });
}

/* =====================================================
   4. Submit Remove (CẬP NHẬT NHẬN 3 ẢNH BASE64)
===================================================== */
document.getElementById('removeBtn').onclick = async () => {
    const groups = canvasDetect.getObjects('group');
    const selectedBoxes = [];

    groups.forEach(g => {
        const boxObj = g.getObjects()[0];
        if (boxObj.isSelectedForRemoval) {
            const originalData = detectedObjects[g.dataIndex];
            selectedBoxes.push({
                box: originalData.box, // Dùng đúng format Pydantic mới [x1, y1, x2, y2]
                label: originalData.label
            });
        }
    });

    if (selectedBoxes.length === 0) return alert("Chưa chọn vật thể nào (Màu Đỏ) để xóa.");

    document.getElementById('loadingOverlay').classList.remove('hidden');

    const payload = { image_name: currentFilename, target_boxes: selectedBoxes };

    try {
        const res = await fetch('http://127.0.0.1:8001/remove_objects', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        // Backend bây giờ phải trả về JSON chứa chuỗi Base64
        const data = await res.json();
        
        if (data.status === 'success') {
            document.getElementById('resultMask').src = "data:image/png;base64," + data.mask_b64;
            document.getElementById('resultLama').src = "data:image/png;base64," + data.lama_b64;
            document.getElementById('resultSDXL').src = "data:image/png;base64," + data.sdxl_b64;
            setTimeout(fitToScreen, 100);
        } else {
            alert("Lỗi khi xóa: " + data.detail);
        }
    } catch (e) { console.error(e); alert("Lỗi kết nối API Backend"); } 
    finally { document.getElementById('loadingOverlay').classList.add('hidden'); }
};