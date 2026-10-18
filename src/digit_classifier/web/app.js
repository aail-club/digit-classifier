// Gradio HTML component scope: element, server. No private Gradio DOM selectors.
const $ = (selector) => element.querySelector(selector);
const video = $('.camera-video');
const still = $('.still-frame');
const overlay = $('.overlay');
const ctx = overlay.getContext('2d');
const frame = document.createElement('canvas');
let stream = null, source = null, busy = false, results = false, disposed = false;
let box = null, previewPending = false, previewAt = 0, epoch = 0, animationId = 0;
let device = 'cpu', previewFailures = 0;

function status(message, error = false) {
  $('.status').textContent = message;
  $('.status').classList.toggle('error', error);
}
function stopCamera() {
  epoch++;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  $('.stop-camera').hidden = true;
}
function sourceSize() {
  return source === 'camera' ? [video.videoWidth, video.videoHeight] : [still.width, still.height];
}
function snapshot(format = 'image/png') {
  const [w, h] = sourceSize();
  if (!w || !h) throw new Error('The camera is still starting. Please try again.');
  const scale = Math.min(1, 1280 / Math.max(w, h));
  frame.width = Math.round(w * scale); frame.height = Math.round(h * scale);
  frame.getContext('2d').drawImage(source === 'camera' ? video : still, 0, 0, frame.width, frame.height);
  return frame.toDataURL(format, 0.94);
}
function drawOverlay() {
  const bounds = overlay.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const w = Math.round(bounds.width * ratio), h = Math.round(bounds.height * ratio);
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  if (!source || results) return;
  const [sw, sh] = sourceSize();
  if (!sw || !sh) return;
  const scale = Math.min(bounds.width / sw, bounds.height / sh);
  const x = (bounds.width - sw * scale) / 2, y = (bounds.height - sh * scale) / 2;
  const size = Math.floor(Math.min(sw, sh) * .6);
  ctx.strokeStyle = '#f4c95d'; ctx.lineWidth = 3;
  ctx.strokeRect(x + Math.floor((sw - size) / 2) * scale, y + Math.floor((sh - size) / 2) * scale, size * scale, size * scale);
  if (box) {
    ctx.strokeStyle = '#50b4ff'; ctx.lineWidth = 2;
    ctx.strokeRect(x + box[0] * sw * scale, y + box[1] * sh * scale, box[2] * sw * scale, box[3] * sh * scale);
  }
}
async function detect() {
  if (previewPending || !source || busy || results || (source === 'camera' && video.readyState < 2)) return;
  const requestEpoch = epoch;
  previewPending = true;
  try {
    const result = await server.preview_frame(snapshot('image/jpeg'));
    if (requestEpoch === epoch && !busy) box = result.box;
    previewFailures = 0;
  } catch (error) {
    box = null;
    if (++previewFailures === 3) status('Detection guide unavailable. You can still capture a digit.', true);
  } finally { previewPending = false; }
}
function tick(now) {
  if (disposed) return;
  if (!results) drawOverlay();
  if (now - previewAt > 400 && document.visibilityState === 'visible') { previewAt = now; detect(); }
  animationId = requestAnimationFrame(tick);
}
async function startCamera() {
  if (busy || disposed) return;
  stopCamera();
  status('Requesting camera access…');
  $('.start-camera').disabled = true;
  const requestEpoch = epoch;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost. You can upload a photo instead.');
    const next = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: 'environment' }, audio: false });
    if (disposed || requestEpoch !== epoch) { next.getTracks().forEach(t => t.stop()); return; }
    stream = next;
    video.srcObject = stream;
    await video.play();
    source = 'camera'; box = null;
    video.hidden = false; still.hidden = true;
    $('.camera-placeholder').hidden = true;
    $('.resume-camera').hidden = true;
    $('.stop-camera').hidden = false;
    $('.capture').disabled = false;
    status(`Ready | Device: ${device}`);
  } catch (error) {
    if (requestEpoch !== epoch) return;
    stopCamera();
    if (source !== 'upload') { source = null; $('.camera-placeholder').hidden = false; $('.capture').disabled = true; }
    const messages = {NotAllowedError:'Camera permission was denied. Allow camera access in your browser, or upload a photo.', NotFoundError:'No camera was found. Connect a camera or upload a photo.', NotReadableError:'The camera is busy. Close other camera apps and try again.'};
    status(messages[error.name] || error.message, true);
  } finally { $('.start-camera').disabled = false; }
}

// Bounded Catmull–Rom control points match the Qt curve, using actual probabilities.
function probabilitySVG(values, digit) {
  const left = 54, top = 28, width = 410, height = 215, bottom = top + height;
  const points = values.map((v, i) => [left + i * width / 9, bottom - v * height]);
  const bounded = ([x,y]) => [Math.min(left+width, Math.max(left,x)), Math.min(bottom,Math.max(top,y))];
  let path = `M ${points[0].join(' ')}`;
  for (let i=0;i<9;i++) {
    const p0=points[Math.max(0,i-1)], p1=points[i], p2=points[i+1], p3=points[Math.min(9,i+2)];
    const c1=bounded([p1[0]+(p2[0]-p0[0])/6,p1[1]+(p2[1]-p0[1])/6]);
    const c2=bounded([p2[0]-(p3[0]-p1[0])/6,p2[1]-(p3[1]-p1[1])/6]);
    path += ` C ${c1.join(' ')} ${c2.join(' ')} ${p2.join(' ')}`;
  }
  const grid = [0,50,100].map(v=>{const y=bottom-v/100*height;return `<line x1="${left}" y1="${y}" x2="${left+width}" y2="${y}" stroke="#4f9b9b" opacity=".22"/><text x="8" y="${y+4}" fill="#929db2">${v}%</text>`;}).join('');
  const dots = points.map(([x,y],i)=>`<circle cx="${x}" cy="${y}" r="${i===digit?6:4}" fill="${i===digit?'#f4c95d':'#4f9b9b'}"/><text x="${x}" y="${bottom+25}" text-anchor="middle" fill="${i===digit?'#f4c95d':'#929db2'}">${i}</text>`).join('');
  const [x,y]=points[digit], labelY=y<top+24?y+24:y-14;
  return `<svg viewBox="0 0 490 290" role="img" aria-label="Probability for digits zero through nine" style="font:12px Inter,Segoe UI,sans-serif">${grid}<path d="M ${left} ${top} V ${bottom} H ${left+width}" fill="none" stroke="#4f9b9b"/><path d="${path}" fill="none" stroke="#4f9b9b" stroke-width="3"/>${dots}<text x="${x}" y="${labelY}" text-anchor="middle" fill="#f4c95d" font-weight="700">${(values[digit]*100).toFixed(1)}%</text></svg>`;
}
function networkSVG(digit) {
  // This is explicitly an architecture illustration, not simulated activations.
  const layers = [[50,5],[195,4],[335,5],[485,3]];
  let lines='', nodes='';
  layers.forEach(([x,n],l)=>{
    for(let i=0;i<n;i++) {
      const y=55+i*130/(n-1);
      nodes+=`<circle class="signal" style="animation-delay:${l*.4}s" cx="${x}" cy="${y}" r="${l===3?13:8}" fill="#030712" stroke="${l===3?'#f4c95d':'#4f9b9b'}" stroke-width="2"/>`;
      if(l<3) { const [nx,nn]=layers[l+1]; for(let j=0;j<nn;j++) lines+=`<line class="signal" style="animation-delay:${l*.4}s" x1="${x+9}" y1="${y}" x2="${nx-9}" y2="${55+j*130/(nn-1)}" stroke="#4f9b9b" opacity=".3"/>`; }
    }
  });
  return `<svg viewBox="0 0 540 240" aria-label="CNN architecture illustration" role="img">${lines}${nodes}<g fill="#929db2" font-size="12" text-anchor="middle"><text x="50" y="220">28 × 28 input</text><text x="195" y="220">Convolutions</text><text x="335" y="220">Classifier</text><text x="485" y="220">Digit ${digit}</text></g></svg><p class="fallback-note">Architecture illustration · digit-specific animation not installed</p>`;
}
function showResult(result) {
  $('.prediction').textContent = String(result.digit);
  $('.confidence').textContent = result.confidence;
  $('.probability-curve').innerHTML = probabilitySVG(result.probabilities, result.digit);
  $('.probability-table tbody').replaceChildren(...result.probabilities.map((p,i)=>{
    const row=document.createElement('tr');
    for(const text of [String(i),`${(p*100).toFixed(2)}%`]) {const cell=document.createElement('td');cell.textContent=text;row.append(cell);} return row;
  }));
  const gif=$('.network-gif'), fallback=$('.network-fallback');
  gif.hidden=!result.animation; fallback.hidden=Boolean(result.animation);
  if(result.animation) gif.src=result.animation;
  else {gif.removeAttribute('src');fallback.innerHTML=networkSVG(result.digit);}
  results=true;
  stopCamera();
  $('.camera-screen').hidden=true;
  $('.results-screen').hidden=false;
  $('.results-header h2').focus({preventScroll:true});
}
$('.capture').addEventListener('click', async () => {
  if (busy || !source) return;
  busy=true; epoch++; $('.capture').disabled=true; $('.upload-button').disabled=true;
  status('Processing…');
  try {
    const data=snapshot();
    // Freeze the precise submitted frame, without drawing overlays into model input.
    still.width=frame.width; still.height=frame.height;
    still.getContext('2d').drawImage(frame,0,0);
    video.hidden=true; still.hidden=false;
    const response=await server.classify_frame(data);
    if (disposed) return;
    if(response.error) throw new Error(response.error);
    device=response.device;
    showResult(response);
  } catch(error) {
    status(error.message || 'Could not classify this frame. Please try again.',true);
    if(source==='camera') {video.hidden=false;still.hidden=true;}
  } finally {busy=false;$('.capture').disabled=!source;$('.upload-button').disabled=false;}
});
$('.try-again').addEventListener('click', () => {
  results=false;box=null;
  $('.results-screen').hidden=true;$('.camera-screen').hidden=false;
  $('.network-gif').removeAttribute('src');
  if(source==='camera') { source=null;$('.capture').disabled=true;startCamera(); }
  else { status(`Ready | Device: ${device}`);$('.capture').focus({preventScroll:true}); }
});
$('.start-camera').addEventListener('click',startCamera);
$('.resume-camera').addEventListener('click',startCamera);
$('.stop-camera').addEventListener('click',()=>{
  if(busy) return;
  stopCamera(); source=null; box=null; $('.capture').disabled=true;
  still.hidden=true; video.hidden=false; $('.camera-placeholder').hidden=false;
  status('Camera is off. Enable your camera or upload a photo.');
});
$('.upload-button').addEventListener('click',()=>$('.file-input').click());
$('.file-input').addEventListener('change', async event => {
  const file=event.target.files[0]; event.target.value='';
  if(!file || busy) return;
  if(file.size>8*1024*1024) {status('Choose an image smaller than 8 MB.',true);return;}
  const url=URL.createObjectURL(file);
  try {
    const image=new Image();image.src=url;await image.decode();
    if(Math.min(image.width,image.height)<32) throw new Error('Choose an image at least 32 pixels wide and tall.');
    stopCamera();epoch++;source='upload';box=null;
    const scale=Math.min(1,1280/Math.max(image.width,image.height));
    still.width=Math.round(image.width*scale);still.height=Math.round(image.height*scale);
    still.getContext('2d').drawImage(image,0,0,still.width,still.height);
    still.hidden=false;video.hidden=true;$('.camera-placeholder').hidden=true;
    $('.resume-camera').hidden=false;$('.capture').disabled=false;
    status(`Ready | Device: ${device}`);
  } catch(error) {status(error.message || 'Could not read the image. Choose a JPEG or PNG.',true);}
  finally {URL.revokeObjectURL(url);}
});

server.app_status().then(result=>{
  device=result.device;
  if(!result.ready) {$('.model-notice').textContent=result.message;$('.model-notice').hidden=false;}
}).catch(()=>status('Could not connect to the server. Refresh to try again.',true));
animationId=requestAnimationFrame(tick);
function cleanup() {disposed=true;cancelAnimationFrame(animationId);stopCamera();observer.disconnect();window.removeEventListener('pagehide',cleanup);}
const observer=new MutationObserver(()=>{if(!element.isConnected) cleanup();});
observer.observe(document.body,{childList:true,subtree:true});
window.addEventListener('pagehide',cleanup);
