'use strict';
const $ = id => document.getElementById(id);
let current = null, mask, instances, original, selectedLabel = 1, dirty = false, drawing = false;
let history = [], polygon = [], lastPoint = null, items = [], loading = false;
let lastKind='video',lastFilter='all';
const canvas = $('canvas'), context = canvas.getContext('2d');
const colors = {0:[0,0,0], 1:[241,72,77], 2:[38,219,229], 255:[177,103,232]};
const states = {visible:'正常可见', partial_occlusion:'部分遮挡', occluded:'完全遮挡', lost:'失跟', out_of_view:'离开视野', split_pending:'分裂待确认'};
const names = {pending:'待审', approved:'已审核', rejected:'已排除'};
const flagNames={flicker:'跨帧闪烁',wall_false_positive:'腔壁误检',fiber_occlusion:'光纤遮挡',
 no_stone_annotation_not_verified_negative:'缺少结石标注，待核实是否为负样本',
 fiber_absence_or_completeness_unverified:'光纤存在性或标注完整性待核实',
 laser_spot_without_fiber_annotation:'有激光光斑，未标注光纤','unsupported_class:0':'含未知原始类别'};
function busy(value){for(const id of ['kind','filter','items','previous','next'])$(id).disabled=value;}
function message(text, error=false) { $('status').textContent=text; $('status').classList.toggle('error',error); }
async function api(path, options) { const response=await fetch(path,options); const data=await response.json(); if(!response.ok) throw new Error(data.error||'请求失败'); return data; }
function fromBase64(value) { const text=atob(value), bytes=new Uint8Array(text.length); for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i); return bytes; }
function toBase64(bytes) { let value=''; for(let i=0;i<bytes.length;i+=8192)value+=String.fromCharCode(...bytes.subarray(i,i+8192)); return btoa(value); }
function markDirty() { dirty=true; message('存在未保存的修订'); }
function checkpoint() { history.push({mask:mask.slice(),instances:instances.slice()}); if(history.length>20)history.shift(); }
function draw() {
 if(!current || !original)return;
 context.clearRect(0,0,canvas.width,canvas.height);context.drawImage(original,0,0);
 if($('overlay').checked) {
  const pixels=context.getImageData(0,0,canvas.width,canvas.height), opacity=Number($('opacity').value)/100;
  for(let i=0;i<mask.length;i++) {
   if(mask[i]===0)continue;
   let color=colors[mask[i]];
   if($('instanceColors').checked && mask[i]===1 && instances[i]>0) { const id=instances[i];color=[70+(id*71)%185,70+(id*113)%185,70+(id*173)%185]; }
   for(let c=0;c<3;c++)pixels.data[i*4+c]=Math.round(pixels.data[i*4+c]*(1-opacity)+color[c]*opacity);
  }
  context.putImageData(pixels,0,0);
 }
 if(polygon.length) {
  context.beginPath();context.moveTo(...polygon[0]);for(const point of polygon.slice(1))context.lineTo(...point);
  context.strokeStyle='#ffffff';context.lineWidth=2;context.stroke();
  for(const p of polygon){context.beginPath();context.arc(p[0],p[1],3,0,Math.PI*2);context.fillStyle='#ffffff';context.fill();}
 }
}
function point(event) { const bounds=canvas.getBoundingClientRect();return [(event.clientX-bounds.left)*canvas.width/bounds.width,(event.clientY-bounds.top)*canvas.height/bounds.height]; }
function writePixel(index) { mask[index]=selectedLabel;instances[index]=selectedLabel===1?Math.max(0,Math.min(65534,Number($('paintId').value)||0)):0; }
function brush(x,y) {
 const radius=Number($('radius').value), w=canvas.width,h=canvas.height;
 for(let py=Math.max(0,Math.floor(y-radius));py<Math.min(h,Math.ceil(y+radius+1));py++)
  for(let px=Math.max(0,Math.floor(x-radius));px<Math.min(w,Math.ceil(x+radius+1));px++)
   if((px-x)**2+(py-y)**2<=radius**2)writePixel(py*w+px);
}
canvas.addEventListener('pointerdown',event=>{
 if(!current||loading)return;event.preventDefault();
 if($('tool').value==='polygon'){polygon.push(point(event));draw();return;}
 checkpoint();drawing=true;canvas.setPointerCapture(event.pointerId);lastPoint=point(event);brush(...lastPoint);markDirty();draw();
});
canvas.addEventListener('pointermove',event=>{
 if(!drawing)return;const p=point(event),distance=Math.hypot(p[0]-lastPoint[0],p[1]-lastPoint[1]), steps=Math.max(1,Math.ceil(distance/Math.max(1,Number($('radius').value)/3)));
 for(let i=1;i<=steps;i++)brush(lastPoint[0]+(p[0]-lastPoint[0])*i/steps,lastPoint[1]+(p[1]-lastPoint[1])*i/steps);
 lastPoint=p;draw();
});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drawing=false);
$('finishPolygon').onclick=()=>{
 if(polygon.length<3){message('多边形至少需要三个点',true);return;}
 checkpoint();const temp=document.createElement('canvas');temp.width=canvas.width;temp.height=canvas.height;const ctx=temp.getContext('2d');
 ctx.beginPath();ctx.moveTo(...polygon[0]);for(const p of polygon.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.fill();const alpha=ctx.getImageData(0,0,temp.width,temp.height).data;
 for(let i=0;i<mask.length;i++)if(alpha[i*4+3]>=128)writePixel(i);polygon=[];markDirty();draw();
};
function undo() {if(polygon.length){polygon=[];draw();return;}if(!history.length)return;({mask,instances}=history.pop());markDirty();draw();}
$('undo').onclick=undo;
document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();undo();}if(['0','1','2','x','X'].includes(event.key))selectLabel(event.key.toLowerCase()==='x'?255:Number(event.key));if(event.key==='Escape'){polygon=[];draw();}});
function selectLabel(label){selectedLabel=label;document.querySelectorAll('[data-label]').forEach(button=>button.classList.toggle('active',Number(button.dataset.label)===label));}
document.querySelectorAll('[data-label]').forEach(button=>button.onclick=()=>selectLabel(Number(button.dataset.label)));
for(const id of ['overlay','opacity','instanceColors'])$(id).addEventListener('input',draw);
$('tool').onchange=()=>{polygon=[];draw();};
function addTrack(value={track_id:1,visibility:'visible',parent_ids:[]}) {
 const box=document.createElement('div');box.className='track';
 const idLabel=document.createElement('label');idLabel.textContent='实例编号';const id=document.createElement('input');id.type='number';id.min=1;id.max=65534;id.value=value.track_id;id.className='trackId';idLabel.append(id);
 const stateLabel=document.createElement('label');stateLabel.textContent='当前状态';const state=document.createElement('select');state.className='trackState';for(const [key,title] of Object.entries(states)){const option=new Option(title,key);state.add(option);}state.value=value.visibility;stateLabel.append(state);
 const parentLabel=document.createElement('label');parentLabel.textContent='碎裂父实例（可空，多个用逗号分隔）';const parent=document.createElement('input');parent.className='parents';parent.value=(value.parent_ids||[]).join(',');parentLabel.append(parent);
 const remove=document.createElement('button');remove.textContent='删除该状态记录';remove.className='remove';remove.onclick=()=>{box.remove();markDirty();};box.append(idLabel,stateLabel,parentLabel,remove);$('tracks').append(box);
 box.addEventListener('change',markDirty);
}
$('addTrack').onclick=()=>{addTrack({track_id:Math.max(1,Number($('paintId').value)||1),visibility:'visible',parent_ids:[]});markDirty();};
function tracks() {return [...$('tracks').children].map(box=>({track_id:Number(box.querySelector('.trackId').value),visibility:box.querySelector('.trackState').value,parent_ids:box.querySelector('.parents').value.split(/[,，]/).map(s=>s.trim()).filter(Boolean).map(Number)}));}
async function load(key) {
 if(loading)return;
 if(dirty&&!window.confirm('当前修订尚未保存。确定放弃并切换样本吗？')){$('items').value=current.id;return;}
 loading=true;busy(true);current=null;original=null;context.clearRect(0,0,canvas.width,canvas.height);message('正在载入样本…');
 try {
  const row=await api('/api/item?id='+encodeURIComponent(key));
  const image=new Image();image.src='/asset?type=image&id='+encodeURIComponent(key);await image.decode();
  current=row;original=image;mask=fromBase64(row.mask_base64);const bytes=fromBase64(row.instances_base64);
  instances=new Uint16Array(bytes.length/2);const view=new DataView(bytes.buffer);for(let i=0;i<instances.length;i++)instances[i]=view.getUint16(i*2,true);
  canvas.width=row.width;canvas.height=row.height;history=[];polygon=[];dirty=false;
  $('title').textContent=row.title;$('reviewState').textContent=row.review_status==='pending'&&row.pixel_review_status==='approved'?'标签已审 · 来源待审':names[row.review_status];$('items').value=key;
  $('approveLabels').hidden=row.kind!=='dataset';$('approve').textContent=row.kind==='dataset'?'批准训练样本':'审核通过并保存';
  $('reviewer').value=row.reviewer||sessionStorage.getItem('reviewer')||'';$('note').value=row.note;$('presence').value=row.stone_presence;$('fiber').value=row.fiber_completeness;
  $('patient').value=row.patient_id;$('sourceVideo').value=row.source_video_id;$('verified').checked=row.source_verified;
  $('flags').textContent=row.flags.map(flag=>flagNames[flag]||'待核实事件').join(' · ')||'尚无事件类别；请结合原图核验';$('sourceHint').textContent='来源线索（未核实）：'+row.source_hint;
  $('sourceFields').hidden=row.kind!=='dataset';$('instanceFields').hidden=row.kind!=='video';$('approveInstances').checked=row.instance_status==='approved';
  $('tracks').replaceChildren();for(const track of row.tracks)addTrack(track);$('paintId').value=0;
  $('preview').removeAttribute('src');$('preview').hidden=true;$('play').hidden=!row.has_preview;
  message(row.is_model_seed?'当前为预标注，尚未审核。请修订并保存。':'已载入保存的修订 · 版本 '+row.revision);draw();
 }catch(error){message(error.message,true);}finally{loading=false;busy(false);}
}
async function refresh(keepCurrent=false) {
 if(!keepCurrent&&dirty){if(!window.confirm('当前修订尚未保存。确定放弃并切换样本吗？')){$('kind').value=lastKind;$('filter').value=lastFilter;return;}dirty=false;}
 busy(true);
 try {
  const result=await api('/api/items?kind='+$('kind').value+'&filter='+$('filter').value);items=result.items;
  $('items').replaceChildren();for(const row of items)$('items').add(new Option(`[${row.review_status==='pending'&&row.pixel_review_status==='approved'?'标签已审 · 来源待审':names[row.review_status]}] ${row.title}`,row.id));
  $('count').textContent=`共 ${items.length} 个样本`;
  lastKind=$('kind').value;lastFilter=$('filter').value;
  if(keepCurrent&&current&&items.some(row=>row.id===current.id))$('items').value=current.id;
  else if(items.length)await load(items[0].id);else {current=null;original=null;context.clearRect(0,0,canvas.width,canvas.height);$('title').textContent='没有符合条件的样本';message('此筛选下没有样本');}
 }catch(error){message(error.message,true);}finally{busy(false);}
}
for(const id of ['kind','filter'])$(id).onchange=()=>refresh();
$('items').onchange=()=>load($('items').value);
for(const [id,offset] of [['previous',-1],['next',1]])$(id).onclick=()=>{const index=items.findIndex(row=>row.id===current?.id), target=items[index+offset];if(target)load(target.id);};
$('play').onclick=()=>{const preview=$('preview');preview.hidden=!preview.hidden;if(!preview.hidden)preview.src='/context?id='+encodeURIComponent(current.id)+'&revision='+Date.now();else preview.removeAttribute('src');};
for(const id of ['reviewer','presence','fiber','patient','sourceVideo','verified','approveInstances','note'])$(id).addEventListener('change',markDirty);
async function save(status, approveLabels=false) {
 if(!current||loading)return;
 if(polygon.length){message('请先完成多边形，或按 Escape 取消',true);return;}
 const bytes=new Uint8Array(instances.length*2),view=new DataView(bytes.buffer);for(let i=0;i<instances.length;i++)view.setUint16(i*2,instances[i],true);
 const payload={id:current.id,revision:current.revision,mask_base64:toBase64(mask),instances_base64:toBase64(bytes),review_status:status,
  reviewer:$('reviewer').value,stone_presence:$('presence').value,fiber_completeness:$('fiber').value,source_verified:$('verified').checked,
  pixel_review_status:status==='approved'||approveLabels?'approved':'pending',
  patient_id:$('patient').value,source_video_id:$('sourceVideo').value,note:$('note').value,
  instance_status:status==='approved'&&$('approveInstances').checked?'approved':'pending',tracks:tracks()};
 loading=true;document.querySelectorAll('.actions button').forEach(b=>b.disabled=true);
 try{const result=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  current.revision=result.revision;current.review_status=result.review_status;dirty=false;sessionStorage.setItem('reviewer',$('reviewer').value);
  $('reviewState').textContent=approveLabels?'标签已审 · 来源待审':names[result.review_status];loading=false;await refresh(true);message(`已保存 · ${approveLabels?'标签已审，来源仍待核实':names[result.review_status]} · 版本 ${result.revision}`);
 }catch(error){message(error.message,true);}finally{loading=false;document.querySelectorAll('.actions button').forEach(b=>b.disabled=false);}
}
$('save').onclick=()=>save('pending');$('approve').onclick=()=>save('approved');$('reject').onclick=()=>save('rejected');
$('approveLabels').onclick=()=>save('pending',true);
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
refresh();
