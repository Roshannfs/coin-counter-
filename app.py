from flask import Flask, render_template_string, request, jsonify, send_file
import cv2
import numpy as np
import base64
import io
import math
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ---------- Detection (same lightweight logic as before) ----------
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 10
HOUGH_MAX_RADIUS = 80

def detect_circles_hough(gray):
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    return [(int(x), int(y), int(r)) for (x, y, r) in circles]

def detect_circles_contours(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append((int(x), int(y), int(r)))
    return circles

def annotate_image(img, markers):
    out = img.copy()
    for i, (x, y, r) in enumerate(markers, start=1):
        cv2.circle(out, (int(x), int(y)), int(r), (30, 144, 255), 2)
        cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(out, str(i), (int(x)-10, int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(out, (10, 10), (180, 50), (0,0,0), -1)
    cv2.putText(out, f"Count: {len(markers)}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    return out

# helper: image to base64
def img_to_base64(img):
    _, buf = cv2.imencode('.png', img)
    b64 = base64.b64encode(buf).decode('ascii')
    return 'data:image/png;base64,' + b64

# ---------- Flask routes ----------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Lightweight Coin Counter - Back Camera</title>
  <style>
    body{font-family: Arial, Helvetica, sans-serif; margin:20px}
    #preview{border:1px solid #ccc}
    canvas{cursor: crosshair; border:1px solid #999}
    table{border-collapse: collapse; margin-top:10px}
    th,td{border:1px solid #ccc; padding:6px}
    #controls{margin-top:10px}
    #videoWrap{display:inline-block; vertical-align:top; margin-right:20px}
  </style>
</head>
<body>
  <h2>Lightweight Coin Counter (Web) - Back Camera Preferred</h2>
  <div>
    <input type="file" id="fileInput" accept="image/*">
    <button id="detectBtn">Detect</button>
    <button id="downloadBtn">Download Logs (.xlsx)</button>
  </div>
  <div id="controls">Left-click: add marker &nbsp; Right-click: delete nearest marker</div>

  <div style="margin-top:12px">
    <div id="videoWrap">
      <video id="video" width="320" height="240" autoplay muted playsinline style="border:1px solid #ccc"></video><br>
      <button id="openCamera">Open Camera</button>
      <button id="captureBtn">Capture</button>
      <label><input type="checkbox" id="autoDetect" checked> Auto-detect on capture</label>
    </div>
    <canvas id="canvas" width="600" height="800"></canvas>
  </div>

  <div id="logArea">
    <h3>Markers / Logs</h3>
    <table id="logTable"><thead><tr><th>#</th><th>X</th><th>Y</th><th>R</th></tr></thead><tbody></tbody></table>
  </div>

<script>
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let img = new Image();
let markers = []; // {x,y,r}
let scale = 1;
let img_w=0,img_h=0;
let video = document.getElementById('video');
let stream = null;

function fitCanvasToImage(w,h){
  const maxDim = 900;
  let sw = w, sh = h;
  if(Math.max(w,h) > maxDim){
    let s = maxDim / Math.max(w,h);
    sw = Math.round(w*s); sh = Math.round(h*s);
  }
  canvas.width = sw; canvas.height = sh;
  scale = sw / w;
}

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(img.src){
    ctx.drawImage(img,0,0, canvas.width, canvas.height);
  }
  markers.forEach((m,i)=>{
    ctx.beginPath();
    ctx.strokeStyle = 'rgb(30,144,255)';
    ctx.arc(m.x*scale, m.y*scale, m.r*scale, 0, Math.PI*2);
    ctx.stroke();
    ctx.fillStyle='red'; ctx.fillRect(m.x*scale-2, m.y*scale-2,4,4);
    ctx.fillStyle='white'; ctx.font='14px Arial';
    ctx.fillText((i+1).toString(), m.x*scale-10, m.y*scale+12);
  });
}

function updateLogTable(){
  let tbody = document.querySelector('#logTable tbody');
  tbody.innerHTML='';
  markers.forEach((m,i)=>{
    let tr = document.createElement('tr');
    tr.innerHTML = `<td>${i+1}</td><td>${Math.round(m.x)}</td><td>${Math.round(m.y)}</td><td>${Math.round(m.r)}</td>`;
    tbody.appendChild(tr);
  });
}

canvas.addEventListener('contextmenu', e=>e.preventDefault());
canvas.addEventListener('mousedown', function(e){
  if(!img.src) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / scale; // natural coords
  const y = (e.clientY - rect.top) / scale;
  if(e.button === 0){ // left -> add
    markers.push({x:x,y:y,r:30});
    updateLogTable(); draw();
  } else if(e.button === 2){ // right -> delete nearest
    if(markers.length===0) return;
    let nearest = 0; let nd = Infinity;
    for(let i=0;i<markers.length;i++){
      let dx = markers[i].x - x; let dy = markers[i].y - y; let d = Math.hypot(dx,dy);
      if(d<nd){ nd=d; nearest=i; }
    }
    if(nd < 50){ markers.splice(nearest,1); updateLogTable(); draw(); }
  }
});

// handle file upload + detect
async function detectFile(file){
  const fd = new FormData(); fd.append('image', file);
  const resp = await fetch('/detect', {method:'POST', body:fd});
  const data = await resp.json();
  if(data.error){ alert(data.error); return; }
  img.src = data.annotated; img.onload = ()=>{
    img_w = data.width; img_h = data.height;
    fitCanvasToImage(img_w, img_h);
    markers = data.markers.map(m=>({x:m[0], y:m[1], r:m[2]}));
    updateLogTable(); draw();
  };
}

document.getElementById('detectBtn').addEventListener('click', async ()=>{
  const f = document.getElementById('fileInput').files[0];
  if(!f){ alert('Choose an image first'); return; }
  await detectFile(f);
});

// download logs
document.getElementById('downloadBtn').addEventListener('click', async ()=>{
  if(markers.length===0){ alert('No markers to download'); return; }
  const resp = await fetch('/download', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({markers:markers})
  });
  if(!resp.ok){ alert('Failed to get file'); return; }
  const blob = await resp.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'coin_logs.xlsx'; document.body.appendChild(a); a.click(); a.remove();
  window.URL.revokeObjectURL(url);
});

// --- Robust back-camera controls ---
async function stopStream(){
  if(stream){
    try{ stream.getTracks().forEach(t=>t.stop()); }catch(e){}
    stream = null; video.srcObject = null;
  }
}

async function openBackCamera(){
  // stop any existing stream
  await stopStream();
  // 1) try facingMode ideal (simple)
  try{
    const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
    // attach
    stream = s; video.srcObject = stream; return;
  }catch(err){
    console.warn('facingMode failed, falling back to device selection', err);
  }

  // 2) enumerate devices and pick a label that suggests back camera
  try{
    let devices = await navigator.mediaDevices.enumerateDevices();
    let videoInputs = devices.filter(d => d.kind === 'videoinput');
    // If labels are empty (no permission), request a quick stream to get labels
    if(videoInputs.length>0 && videoInputs[0].label === ''){
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach(t=>t.stop());
      devices = await navigator.mediaDevices.enumerateDevices();
      videoInputs = devices.filter(d => d.kind === 'videoinput');
    }
    if(videoInputs.length === 0) throw new Error('No video input devices');

    // prefer labels containing keywords
    let preferred = videoInputs.find(d => /back|rear|environment|main/i.test(d.label));
    if(!preferred) preferred = videoInputs[0];
    const deviceId = preferred.deviceId;
    stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId } }, audio: false });
    video.srcObject = stream;
  }catch(err){
    console.error('openBackCamera failed:', err); alert('Could not open back camera: '+err.message);
  }
}

// Bind open camera button
document.getElementById('openCamera').addEventListener('click', ()=>{
  openBackCamera();
});

// Capture button: capture frame and optionally auto-detect
document.getElementById('captureBtn').addEventListener('click', async ()=>{
  if(!stream){ alert('Open the camera first'); return; }
  const off = document.createElement('canvas'); off.width = video.videoWidth; off.height = video.videoHeight;
  const offCtx = off.getContext('2d');
  offCtx.drawImage(video, 0, 0, off.width, off.height);
  off.toBlob(async (blob)=>{
    const url = URL.createObjectURL(blob);
    img.src = url; img.onload = ()=>{ img_w = off.width; img_h = off.height; fitCanvasToImage(img_w, img_h); markers = []; draw(); };
    if(document.getElementById('autoDetect').checked){
      const fd = new FormData(); fd.append('image', blob, 'capture.png');
      const resp = await fetch('/detect', {method:'POST', body:fd});
      const data = await resp.json();
      if(data.error){ alert(data.error); return; }
      img.src = data.annotated; img.onload = ()=>{ img_w = data.width; img_h = data.height; fitCanvasToImage(img_w, img_h); markers = data.markers.map(m=>({x:m[0], y:m[1], r:m[2]})); updateLogTable(); draw(); };
    }
  }, 'image/png');
});

</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/detect', methods=['POST'])
def detect_route():
    if 'image' not in request.files:
        return jsonify({'error':'No image uploaded'}), 400
    file = request.files['image']
    in_bytes = file.read()
    arr = np.frombuffer(in_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error':'Cannot decode image'}), 400
    # resize large images for performance
    max_dim = 900
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    markers = detect_circles_hough(gray)
    if not markers:
        markers = detect_circles_contours(gray)
    annotated = annotate_image(img, markers)
    b64 = img_to_base64(annotated)
    return jsonify({'annotated': b64, 'markers': markers, 'width': w, 'height': h})

@app.route('/download', methods=['POST'])
def download_route():
    data = request.get_json()
    if not data or 'markers' not in data:
        return jsonify({'error':'No markers provided'}), 400
    markers = data['markers']
    df = pd.DataFrame(markers, columns=['x','y','r'])
    df.insert(0, 'id', range(1, len(df)+1))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='markers')
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f'coin_logs_{ts}.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    # Note: for HTTPS in development you can provide SSL context to app.run
    # e.g. app.run(ssl_context=('cert.pem','key.pem'))
    app.run(debug=True)
