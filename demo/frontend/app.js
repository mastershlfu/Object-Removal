/* =====================================================
   Globals & Init
===================================================== */
const canvasROI = new fabric.Canvas('canvasROI', { selection: false, backgroundColor: '#000' });
const canvasDetect = new fabric.Canvas('canvasDetect', { selection: true, backgroundColor: '#000' });

let currentFilename = "";
let imgOriginalWidth = 0;
let imgOriginalHeight = 0;

let drawMode = false;
let isDrawing = false;
let rectROI = null;
let startX, startY;

let detectedObjects = []; 
let viewScale = 1.0; 

// Cấu hình Fabric Handle
fabric.Object.prototype.set({
    cornerColor: '#00e5ff', cornerStyle: 'circle', transparentCorners: false
});

/* =====================================================
   1. Upload & Fix Firefox Scale Issue
===================================================== */
document.getElementById('upload').onchange = function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    currentFilename = file.name;
    const reader = new FileReader();
    
    reader.onload = (f) => {
        // Sử dụng Image Object native thay vì fabric.fromURL để kiểm soát tốt hơn
        const imgObj = new Image();
        imgObj.src = f.target.result;
        
        imgObj.onload = () => {
            imgOriginalWidth = imgObj.width;
            imgOriginalHeight = imgObj.height;

            // Setup 2 Canvas
            setupSingleCanvas(canvasROI, imgObj);
            setupSingleCanvas(canvasDetect, imgObj);

            // Setup Result Image
            const resImg = document.getElementById('resultImage');
            resImg.src = f.target.result;
            resImg.style.width = `${imgOriginalWidth}px`;
            resImg.style.height = `${imgOriginalHeight}px`;

            // Reset UI
            document.getElementById('classFilterSection').classList.add('hidden');
            document.getElementById('removeBtn').disabled = true;

            // Gọi Scale (Dùng setTimeout để chờ DOM render layout xong)
            setTimeout(fitToScreen, 50);
        }
    };
    reader.readAsDataURL(file);
    this.value = ''; // Reset input để cho phép upload lại file cũ
};

function setupSingleCanvas(fabricCanvas, imgElement) {
    fabricCanvas.setDimensions({ width: imgOriginalWidth, height: imgOriginalHeight });
    fabricCanvas.clear();
    
    const fImg = new fabric.Image(imgElement, {
        selectable: false, evented: false,
        originX: 'left', originY: 'top'
    });
    fabricCanvas.setBackgroundImage(fImg, fabricCanvas.renderAll.bind(fabricCanvas));
}

// Logic Scale hiển thị (CSS Transform)
function fitToScreen() {
    if (imgOriginalWidth === 0) return;

    const items = [
        { id: 'wrap1', el: canvasROI.getElement().parentElement },
        { id: 'wrap2', el: canvasDetect.getElement().parentElement },
        { id: 'wrap3', el: document.getElementById('resultImage') }
    ];

    items.forEach(item => {
        const wrapper = document.getElementById(item.id);
        if(!wrapper || !item.el) return;

        // Trừ padding 40px
        const scale = Math.min(
            (wrapper.clientWidth - 40) / imgOriginalWidth,
            (wrapper.clientHeight - 40) / imgOriginalHeight
        );

        if(item.id === 'wrap1') viewScale = scale; // Lưu scale để tính nét vẽ

        // Apply Transform căn giữa
        item.el.style.transform = `translate(-50%, -50%) scale(${scale})`;
    });
}

window.addEventListener('resize', () => requestAnimationFrame(fitToScreen));

/* =====================================================
   2. ROI Drawing (Panel 1)
===================================================== */
const drawBtn = document.getElementById('drawBtn');
drawBtn.onclick = () => {
    drawMode = !drawMode;
    drawBtn.innerText = drawMode ? '✏️ Mode: DRAW' : '👁️ Mode: VIEW';
    drawBtn.className = drawMode ? 'btn btn-warning' : 'btn btn-primary';
    
    canvasROI.discardActiveObject();
    canvasROI.requestRenderAll();
    canvasROI.defaultCursor = drawMode ? 'crosshair' : 'default';
};

canvasROI.on('mouse:down', (opt) => {
    if (!drawMode || opt.target) return;
    isDrawing = true;
    const p = canvasROI.getPointer(opt.e);
    startX = p.x; startY = p.y;
    
    rectROI = new fabric.Rect({
        left: startX, top: startY, width: 0, height: 0,
        fill: 'rgba(255, 255, 255, 0.1)',
        stroke: '#00e5ff', 
        strokeWidth: 3 / viewScale, 
        selectable: true, hasControls: true
    });
    canvasROI.add(rectROI);
    canvasROI.setActiveObject(rectROI);
});

canvasROI.on('mouse:move', (opt) => {
    if (!isDrawing) return;
    const p = canvasROI.getPointer(opt.e);
    rectROI.set({
        width: Math.abs(p.x - startX), height: Math.abs(p.y - startY),
        left: Math.min(p.x, startX), top: Math.min(p.y, startY)
    });
    canvasROI.requestRenderAll();
});

canvasROI.on('mouse:up', () => { isDrawing = false; if(rectROI) rectROI.setCoords(); });

document.addEventListener('keydown', (e) => {
    if ((e.key === 'Delete' || e.key === 'Backspace') && canvasROI.getActiveObject()) {
        canvasROI.remove(canvasROI.getActiveObject());
    }
});

/* =====================================================
   3. Scan & Display Logic (Panel 2) - UPDATED
===================================================== */
document.getElementById('scanBtn').onclick = async () => {
    const rois = canvasROI.getObjects('rect');
    if (rois.length === 0) return alert("Please draw ROI first!");

    const payload = {
        image_name: currentFilename,
        boxes: rois.map(r => ({
            xmin: Math.round(r.left), ymin: Math.round(r.top),
            xmax: Math.round(r.left + r.width * r.scaleX), ymax: Math.round(r.top + r.height * r.scaleY)
        }))
    };

    try {
        // Gọi API Mock hoặc Real ở đây
        const res = await fetch('http://127.0.0.1:8001/submit_boxes', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            detectedObjects = data.objects;
            renderDetectedObjects(detectedObjects);
            document.getElementById('removeBtn').disabled = false;
        }
    } catch (err) {
        console.error(err); alert("Error connecting backend");
    }
};

function renderDetectedObjects(objects) {
    canvasDetect.clear();
    // Copy background
    canvasROI.backgroundImage.clone(bg => {
        canvasDetect.setBackgroundImage(bg, canvasDetect.renderAll.bind(canvasDetect));
    });

    const classes = {};

    objects.forEach((obj, index) => {
        // Tạo Box
        const box = new fabric.Rect({
            left: obj.box[0], top: obj.box[1],
            width: obj.box[2] - obj.box[0], height: obj.box[3] - obj.box[1],
            fill: 'rgba(239, 68, 68, 0.4)', // Mặc định ĐỎ (Selected)
            stroke: '#ef4444',
            strokeWidth: 3 / viewScale,
            strokeDashArray: null,
            objectCaching: false,
            // Custom Props
            isSelectedForRemoval: true, 
            classLabel: obj.label
        });

        // Tạo Label Text
        const text = new fabric.Text(obj.label, {
            left: obj.box[0], top: obj.box[1] - (20/viewScale),
            fontSize: 14 / viewScale, fill: '#ef4444',
            backgroundColor: 'rgba(0,0,0,0.8)', fontFamily: 'Inter'
        });

        const group = new fabric.Group([box, text], {
            selectable: true, hasControls: false, lockMovementX: true, lockMovementY: true,
            dataIndex: index, className: obj.label,
            hoverCursor: 'pointer'
        });

        // Click vào box để toggle state
        group.on('mousedown', () => toggleSelectionState(group));

        canvasDetect.add(group);

        // Count class
        if (!classes[obj.label]) classes[obj.label] = 0;
        classes[obj.label]++;
    });

    renderClassFilter(classes);
    canvasDetect.requestRenderAll();
}

/**
 * Hàm Toggle trạng thái Box
 * @param {fabric.Group} groupObj - Group chứa Box và Text
 * @param {boolean|null} forceState - Nếu null thì đảo trạng thái, nếu true/false thì ép trạng thái
 */
function toggleSelectionState(groupObj, forceState = null) {
    const box = groupObj.getObjects()[0];
    const text = groupObj.getObjects()[1];

    // Logic đảo trạng thái
    const newState = forceState !== null ? forceState : !box.isSelectedForRemoval;
    box.isSelectedForRemoval = newState;

    if (newState) {
        // STATE: SELECTED (REMOVE) -> RED
        box.set({ 
            stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.4)', 
            strokeDashArray: null, strokeWidth: 3 / viewScale 
        });
        text.set({ fill: '#ef4444' });
    } else {
        // STATE: UNSELECTED (KEEP) -> GREEN/YELLOW
        box.set({ 
            stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.1)', 
            strokeDashArray: [5, 5], strokeWidth: 2 / viewScale 
        });
        text.set({ fill: '#10b981' });
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
        
        // Mặc định Checked (vì mặc định box là Red)
        div.innerHTML = `
            <input type="checkbox" id="chk-${cls}" checked />
            <label for="chk-${cls}">${cls}</label>
            <span class="badge">${classCounts[cls]}</span>
        `;
        
        // Handle Checkbox Change
        const checkbox = div.querySelector('input');
        checkbox.onchange = (e) => {
            const isChecked = e.target.checked;
            // Tìm tất cả các box thuộc class này và ép trạng thái theo checkbox
            canvasDetect.getObjects('group').forEach(g => {
                if (g.className === cls) {
                    toggleSelectionState(g, isChecked);
                }
            });
        };

        container.appendChild(div);
    });
}

/* =====================================================
   4. Submit Remove (Panel 2 -> Panel 3)
===================================================== */
document.getElementById('removeBtn').onclick = async () => {
    const groups = canvasDetect.getObjects('group');
    const selectedBoxes = [];

    groups.forEach(g => {
        const boxObj = g.getObjects()[0];
        // Chỉ lấy những box có trạng thái là SELECTED (Red)
        if (boxObj.isSelectedForRemoval) {
            const originalData = detectedObjects[g.dataIndex];
            selectedBoxes.push({
                xmin: originalData.box[0], ymin: originalData.box[1],
                xmax: originalData.box[2], ymax: originalData.box[3],
                label: originalData.label
            });
        }
    });

    if (selectedBoxes.length === 0) {
        return alert("No boxes selected (Red) for removal.");
    }

    document.getElementById('loadingOverlay').classList.remove('hidden');

    const payload = {
        image_name: currentFilename,
        target_boxes: selectedBoxes
    };

    try {
        const res = await fetch('http://127.0.0.1:8001/remove_objects', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            document.getElementById('resultImage').src = url;
            
            // Trigger fit screen lại để ảnh result hiển thị đúng
            setTimeout(fitToScreen, 100);
        } else {
            alert("Remove failed.");
        }
    } catch (e) {
        console.error(e); alert("Error API");
    } finally {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }
};