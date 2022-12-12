(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[56],{1021:function(e,t,n){"use strict";var r=n(5793);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function a(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{null!=e&&r.util.assert("complex64"!==e.dtype,()=>`${t} does not support complex64 tensors in the CPU backend.`)})}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let s=r.kernel_impls.whereImpl;class i extends r.KernelBackend{constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new r.DataStorage(this,(0,r.engine)())}nextDataId(){return i.nextDataId++}write(e,t,n){this.firstUse&&(this.firstUse=!1,(0,r.env)().get("IS_NODE")&&r.backend_util.warn("\n============================\nHi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. \n============================"));let a={id:this.nextDataId()};return this.data.set(a,{values:e,dtype:n,refCount:1}),a}makeTensorInfo(e,t,n){let a;if("string"===t&&null!=n&&n.length>0&&r.util.isString(n[0])){let s=n.map(e=>r.util.encodeString(e));a=this.write(s,e,t)}else a=this.write(n,e,t);return{dataId:a,shape:e,dtype:t}}refCount(e){if(this.data.has(e)){let t=this.data.get(e);return t.refCount}return 0}incRef(e){let t=this.data.get(e);t.refCount++}decRef(e){if(this.data.has(e)){let t=this.data.get(e);t.refCount--}}move(e,t,n,r,a){this.data.set(e,{values:t,dtype:r,refCount:a})}numDataIds(){return this.data.numDataIds()}async read(e){return this.readSync(e)}readSync(e){let{dtype:t,complexTensorInfos:n}=this.data.get(e);if("complex64"===t){let a=this.readSync(n.real.dataId),s=this.readSync(n.imag.dataId);return r.backend_util.mergeRealAndImagArrays(a,s)}return this.data.get(e).values}bufferSync(e){let t=this.readSync(e.dataId);if("string"===e.dtype)try{let n=t.map(e=>r.util.decodeString(e));return(0,r.buffer)(e.shape,e.dtype,n)}catch(a){throw Error("Failed to decode encoded string bytes into utf-8")}return(0,r.buffer)(e.shape,e.dtype,t)}makeOutput(e,t,n){return(0,r.engine)().makeTensorFromTensorInfo(this.makeTensorInfo(t,n,e),this)}disposeData(e,t=!1){if(this.data.has(e)){if(this.data.get(e).refCount--,!t&&this.data.get(e).refCount>0)return!1;let{complexTensorInfos:n}=this.data.get(e);null!=n&&(this.disposeData(n.real.dataId,!0),this.disposeData(n.imag.dataId,!0)),this.data.delete(e)}return!0}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}async time(e){let t=r.util.now();e();let n=r.util.now()-t;return{kernelMs:n}}memory(){return{unreliable:!0,reasons:["The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less."]}}where(e){a([e],"where");let t=this.readSync(e.dataId);return s(e.shape,t)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function o(e,t,n){return({inputs:s,attrs:i,backend:o})=>{let{x:u}=s;if(a(u,e),"string"===u.dtype||"string"===n)throw Error("unaryKernelFunc does not support string input/output");let l=o.data.get(u.dataId).values,p=r.util.sizeFromShape(u.shape),c=n||u.dtype,h=r.util.getArrayFromDType(c,p);for(let d=0;d<p;++d)h[d]=t(l[d],i);return o.makeTensorInfo(u.shape,c,h)}}function u(e,t,n){return({inputs:r,attrs:s,backend:i})=>{let{x:o}=r;if(a(o,e),"string"===o.dtype||"string"===n)throw Error("unaryKernelFunc does not support string input/output");let u=i.data.get(o.dataId).values,l=n||o.dtype,p=t(u,l,s);return i.makeTensorInfo(o.shape,l,p)}}i.nextDataId=0,/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ (0,r.registerBackend)("cpu",()=>new i,1);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let l=o(r.Elu,e=>e>=0?e:Math.exp(e)-1),p={kernelName:r.Elu,backendName:"cpu",kernelFunc:l};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function c(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}let h={kernelName:r.Identity,backendName:"cpu",kernelFunc:c};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function d(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{alpha:o}=s;a([i],"leakyRelu");let u=r.util.sizeFromShape(i.shape),l=n.data.get(i.dataId).values,p=r.util.getTypedArrayFromDType("float32",u);for(let c=0;c<l.length;c++)p[c]=l[c]<0?o*l[c]:l[c];return n.makeTensorInfo(i.shape,"float32",p)}let f={kernelName:r.LeakyRelu,backendName:"cpu",kernelFunc:d};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function m(e){return(t,n,a,s,i)=>{let o=r.backend_util.assertAndGetBroadcastShape(t,n),u=o.length,l=r.util.computeStrides(o),p=r.util.sizeFromShape(o),c=r.util.getTypedArrayFromDType(i,p),h=t.length,d=n.length,f=r.util.computeStrides(t),m=r.util.computeStrides(n),g=r.backend_util.getBroadcastDims(t,o),y=r.backend_util.getBroadcastDims(n,o);if(g.length+y.length===0)for(let b=0;b<c.length;++b)c[b]=e(a[b%a.length],s[b%s.length]);else for(let k=0;k<c.length;++k){let N=r.util.indexToLoc(k,u,l),v=N.slice(-h);g.forEach(e=>v[e]=0);let x=r.util.locToIndex(v,h,f),w=N.slice(-d);y.forEach(e=>w[e]=0);let T=r.util.locToIndex(w,d,m);c[k]=e(a[x],s[T])}return[c,o]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let g=m((e,t)=>e<0?t*e:e);function y(e){let{inputs:t,backend:n}=e,{x:r,alpha:s}=t;a([r,s],"prelu");let i=n.data.get(r.dataId).values,o=n.data.get(s.dataId).values,[u,l]=g(r.shape,s.shape,i,o,"float32");return n.makeTensorInfo(l,"float32",u)}let b={kernelName:r.Prelu,backendName:"cpu",kernelFunc:y},k=o(r.Relu,e=>Math.max(0,e)),N={kernelName:r.Relu,backendName:"cpu",kernelFunc:k},v=o(r.Relu6,e=>Math.min(Math.max(0,e),6)),x={kernelName:r.Relu6,backendName:"cpu",kernelFunc:v};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function w(e){return(t,n,a)=>{let s=r.util.getTypedArrayFromDType(n,t.length);for(let i=0;i<t.length;++i)s[i]=e(t[i],a);return s}}w(e=>1/(1+Math.exp(-e)));let T=o(r.Sigmoid,e=>1/(1+Math.exp(-e))),S={kernelName:r.Sigmoid,backendName:"cpu",kernelFunc:T};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function I(e,t,n,r,a){if("linear"===n)return c({inputs:{x:t},backend:e});if("relu"===n)return k({inputs:{x:t},backend:e});if("elu"===n)return l({inputs:{x:t},backend:e});if("relu6"===n)return v({inputs:{x:t},backend:e});if("prelu"===n)return y({inputs:{x:t,alpha:r},backend:e});else if("leakyrelu"===n)return d({inputs:{x:t},backend:e,attrs:{alpha:a}});else if("sigmoid"===n)return T({inputs:{x:t},backend:e});throw Error(`Activation ${n} has not been implemented for the CPU backend.`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function _(e){let{inputs:t,backend:n}=e,{real:r,imag:a}=t,s=n.data.get(r.dataId).values,i=n.data.get(a.dataId).values,o=n.makeTensorInfo(r.shape,"complex64"),u=n.data.get(o.dataId);return u.complexTensorInfos={real:n.makeTensorInfo(r.shape,"float32",s),imag:n.makeTensorInfo(a.shape,"float32",i)},o}let E={kernelName:r.Complex,backendName:"cpu",kernelFunc:_};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function A(e,t,n="float32"){if("complex64"===n){let a=A(e,t,"float32"),s=A(e,t,"float32");return _({inputs:{real:a,imag:s},backend:e})}let i=r.util.makeZerosTypedArray(r.util.sizeFromShape(t),n);return e.makeTensorInfo(t,n,i)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function M(e){let{inputs:t,backend:n}=e,{input:r}=t,a=n.data.get(r.dataId).complexTensorInfos.real,s=n.data.get(a.dataId).values;return n.makeTensorInfo(a.shape,a.dtype,s)}let D={kernelName:r.Real,backendName:"cpu",kernelFunc:M};function $(e){let{inputs:t,backend:n,attrs:a}=e,{x:s}=t,{dtype:i}=a;if("complex64"===i){if("complex64"===s.dtype)return c({inputs:{x:s},backend:n});let o=A(n,s.shape,s.dtype),u=$({inputs:{x:s},backend:n,attrs:{dtype:"float32"}}),l=_({inputs:{real:u,imag:o},backend:n});return n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(u),l}if("complex64"===s.dtype){let p=M({inputs:{input:s},backend:n}),h=$({inputs:{x:p},backend:n,attrs:{dtype:i}});return n.disposeIntermediateTensorInfo(p),h}if(!r.util.hasEncodingLoss(s.dtype,i)){let d=c({inputs:{x:s},backend:n});return{dataId:d.dataId,shape:d.shape,dtype:i}}let f=n.data.get(s.dataId).values,[g,y,b]=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a){if("int32"===a){let s=Int32Array.from(e);return[t,"int32",s]}if("bool"===a){let i=r.util.toTypedArray([0],n),[o,u]=m((e,t)=>e!==t?1:0)(t,[],e,i,"bool");return[u,"bool",o]}throw Error(`Error in Cast: failed to cast ${n} to ${a}`)}(f,s.shape,s.dtype,i);return n.makeTensorInfo(g,y,b)}let F={kernelName:r.Cast,backendName:"cpu",kernelFunc:$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function B(e,t,n,s){return null==n?({inputs:n,backend:i})=>{let{a:o,b:u}=n;a([o,u],e);let l=i.data.get(o.dataId).values,p=i.data.get(u.dataId).values,c="string"===o.dtype?r.backend_util.fromUint8ToStringArray(l):l,h="string"===o.dtype?r.backend_util.fromUint8ToStringArray(p):p,d=s||o.dtype,[f,m]=t(o.shape,u.shape,c,h,d);return i.makeTensorInfo(m,d,f)}:({inputs:e,backend:r})=>{let{a,b:i}=e;if("complex64"===a.dtype||"complex64"===i.dtype){let o=$({inputs:{x:a},backend:r,attrs:{dtype:"complex64"}}),u=r.data.get(o.dataId),l=u.complexTensorInfos.real,p=u.complexTensorInfos.imag,c=r.data.get(l.dataId).values,h=r.data.get(p.dataId).values,d=$({inputs:{x:i},backend:r,attrs:{dtype:"complex64"}}),f=r.data.get(d.dataId),m=f.complexTensorInfos.real,g=f.complexTensorInfos.imag,y=r.data.get(m.dataId).values,b=r.data.get(g.dataId).values,[k,N,v]=n(a.shape,i.shape,c,h,y,b),x=r.makeTensorInfo(v,"float32",k),w=r.makeTensorInfo(v,"float32",N),T=_({inputs:{real:x,imag:w},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(d),r.disposeIntermediateTensorInfo(x),r.disposeIntermediateTensorInfo(w),T}{let S=r.data.get(a.dataId).values,I=r.data.get(i.dataId).values,E=s||a.dtype,[A,M]=t(a.shape,i.shape,S,I,E);return r.makeTensorInfo(M,E,A)}}}function O(e){return(t,n,a,s,i,o)=>{let u=r.backend_util.assertAndGetBroadcastShape(t,n),l=r.util.sizeFromShape(u),p=u.length,c=r.util.computeStrides(u),h=r.util.getTypedArrayFromDType("float32",l),d=r.util.getTypedArrayFromDType("float32",l),f=r.backend_util.getBroadcastDims(t,u),m=r.backend_util.getBroadcastDims(n,u),g=r.backend_util.mergeRealAndImagArrays(a,s),y=r.backend_util.mergeRealAndImagArrays(i,o),b=t.length,k=r.util.computeStrides(t),N=n.length,v=r.util.computeStrides(n);if(f.length+m.length===0)for(let x=0;x<h.length;x++){let w=x%g.length,T=x%y.length,S=e(g[2*w],g[2*w+1],y[2*T],y[2*T+1]);h[x]=S.real,d[x]=S.imag}else for(let I=0;I<h.length;I++){let _=r.util.indexToLoc(I,p,c),E=_.slice(-b);f.forEach(e=>E[e]=0);let A=r.util.locToIndex(E,b,k),M=_.slice(-N);m.forEach(e=>M[e]=0);let D=r.util.locToIndex(M,N,v),$=e(g[2*A],g[2*A+1],y[2*D],y[2*D+1]);h[I]=$.real,d[I]=$.imag}return[h,d,u]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let R=m((e,t)=>e+t),C=O((e,t,n,r)=>({real:e+n,imag:t+r})),V=B(r.Add,R,C),P={kernelName:r.Add,backendName:"cpu",kernelFunc:V};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function L(e){let{inputs:t,backend:n,attrs:a}=e,{x:s}=t,{shape:i}=a,o=r.util.sizeFromShape(s.shape),u=r.util.inferFromImplicitShape(i,o),l=r.util.sizeFromShape(u);r.util.assert(o===l,()=>`The new shape (${u}) has ${l} elements and the old shape (${s.shape}) has ${o} elements. The new shape and old shape must have the same number of elements.`),n.incRef(s.dataId);let p=n.data.get(s.dataId);if(null!=p.complexTensorInfos){let c=p.complexTensorInfos.real,h=p.complexTensorInfos.imag;c.shape=u,h.shape=u}return{dataId:s.dataId,shape:u,dtype:s.dtype}}let z={kernelName:r.Reshape,backendName:"cpu",kernelFunc:L};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function W(e){let{inputs:t,backend:n,attrs:s}=e,{a:i,b:o}=t,{transposeA:u,transposeB:l}=s;a([i,o],"matMul");let p=i.shape.length,c=o.shape.length,h=u?i.shape[p-2]:i.shape[p-1],d=l?o.shape[c-1]:o.shape[c-2],f=u?i.shape[p-1]:i.shape[p-2],m=l?o.shape[c-2]:o.shape[c-1],g=i.shape.slice(0,-2),y=o.shape.slice(0,-2),b=r.util.sizeFromShape(g),k=r.util.sizeFromShape(y),N=r.broadcast_util.assertAndGetBroadcastShape(i.shape.slice(0,-2),o.shape.slice(0,-2)),v=N.concat([f,m]);r.util.assert(h===d,()=>`Error in matMul: inner shapes (${h}) and (${d}) of Tensors with shapes ${i.shape} and ${o.shape} and transposeA=${u} and transposeB=${l} must match.`);let x=L({inputs:{x:i},backend:n,attrs:{shape:u?[b,h,f]:[b,f,h]}}),w=L({inputs:{x:o},backend:n,attrs:{shape:l?[k,m,d]:[k,d,m]}}),T=u?x.shape[1]:x.shape[2],S=u?x.shape[2]:x.shape[1],I=l?w.shape[1]:w.shape[2],_=Math.max(b,k),E=n.data.get(x.dataId).values,A=n.data.get(w.dataId).values,M=r.util.computeStrides(x.shape),D=r.util.computeStrides(w.shape),[$,F,B]=u?[M[0],1,M[1]]:[M[0],M[1],1],[O,R,C]=l?[1,D[1],D[0]]:[D[1],1,D[0]],V=S*I,P=(0,r.buffer)([_,S,I],x.dtype),z=P.values,W=n.blockSize;for(let U=0;U<_;U++)for(let G=0;G<S;G+=W)for(let q=0;q<I;q+=W)for(let H=0;H<T;H+=W){let j=Math.min(G+W,S),K=Math.min(q+W,I),X=Math.min(H+W,T);for(let Z=G;Z<j;Z++)for(let Q=q;Q<K;Q++){let Y=0;for(let J=H;J<X;J++){let ee=Math.min(U,b-1)*$,et=Math.min(U,k-1)*C,en=E[ee+Z*F+J*B],er=A[J*O+Q*R+et];Y+=en*er}z[U*V+(Z*I+Q)]+=Y}}return n.disposeIntermediateTensorInfo(x),n.disposeIntermediateTensorInfo(w),n.makeTensorInfo(v,P.dtype,P.values)}let U={kernelName:r.BatchMatMul,backendName:"cpu",kernelFunc:W},G={kernelName:r._FusedMatMul,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:r}=e,{a,b:s,bias:i,preluActivationWeights:o}=t,{transposeA:u,transposeB:l,activation:p,leakyreluAlpha:c}=r,h,d,f,m=[],g=W({inputs:{a,b:s},attrs:{transposeA:u,transposeB:l},backend:n});for(let y of(h=g,i&&(d=V({inputs:{a:h,b:i},backend:n}),m.push(h),h=d),p&&(f=I(n,h,p,o,c),m.push(h),h=f),m))n.disposeIntermediateTensorInfo(y);return h}},q=e=>{let{x:t}=e.inputs,n=e.backend;a(t,"abs");let s=new Float32Array(r.util.sizeFromShape(t.shape)),i=n.data.get(t.dataId).values;return s=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=new Float32Array(e.length);for(let n=0;n<e.length;++n)t[n]=Math.abs(e[n]);return t}(i),n.makeOutput(s,t.shape,t.dtype)},H={kernelName:r.Abs,backendName:"cpu",kernelFunc:q},j=o(r.Acos,e=>Math.acos(e)),K={kernelName:r.Acos,backendName:"cpu",kernelFunc:j},X=o(r.Acosh,e=>Math.acosh(e)),Z={kernelName:r.Acosh,backendName:"cpu",kernelFunc:X},Q={kernelName:r.AddN,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e;a(t,"addN");let s=t.map(e=>n.data.get(e.dataId).values),i=(0,r.buffer)(t[0].shape,t[0].dtype),o=i.values;for(let u=0;u<t.length;u++){let l=s[u];for(let p=0;p<o.length;p++)o[p]+=l[p]}return n.makeTensorInfo(i.shape,i.dtype,i.values)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function Y(e,t,n,a,s){let i=t.length,o=r.util.sizeFromShape(t),u=r.util.computeStrides(t),l=r.util.computeStrides(s),p=r.util.getTypedArrayFromDType(n,r.util.sizeFromShape(s));for(let c=0;c<o;++c){let h=r.util.indexToLoc(c,i,u),d=Array(h.length);for(let f=0;f<d.length;f++)d[f]=h[a[f]];let m=r.util.locToIndex(d,i,l);p[m]=e[c]}return p}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function J(e){let{inputs:t,attrs:n,backend:r}=e,{x:s}=t,{perm:i}=n;a(s,"transpose");let o=s.shape.length,u=Array(o);for(let l=0;l<u.length;l++)u[l]=s.shape[i[l]];let p=r.data.get(s.dataId).values,c=Y(p,s.shape,s.dtype,i,u),h=r.write(c,u,s.dtype);return{dataId:h,shape:u,dtype:s.dtype}}let ee={kernelName:r.Transpose,backendName:"cpu",kernelFunc:J},et={kernelName:r.All,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o,keepDims:u}=s;a(i,"all");let l=r.util.parseAxisParam(o,i.shape),p=l,c=r.backend_util.getAxesPermutation(p,i.shape.length),h=i;null!=c&&(h=J({inputs:{x:i},backend:n,attrs:{perm:c}}),p=r.backend_util.getInnerMostAxes(p.length,i.shape.length)),r.backend_util.assertAxesAreInnerMostDims("all",p,h.shape.length);let[d,f]=r.backend_util.computeOutAndReduceShapes(h.shape,p),m=r.util.sizeFromShape(f),g=r.util.makeZerosTypedArray(r.util.sizeFromShape(d),h.dtype),y=n.data.get(h.dataId).values;for(let b=0;b<g.length;++b){let k=b*m,N=y[k];for(let v=0;v<m;++v){let x=y[k+v];N=N&&x}g[b]=N}null!=c&&n.disposeIntermediateTensorInfo(h);let w=n.makeTensorInfo(d,h.dtype,g);if(u){let T=r.backend_util.expandShapeToKeepDim(d,l),S=L({inputs:{x:w},backend:n,attrs:{shape:T}});return n.disposeIntermediateTensorInfo(w),S}return w}},en={kernelName:r.Any,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o,keepDims:u}=s;a(i,"any");let l=r.util.parseAxisParam(o,i.shape),p=l,c=r.backend_util.getAxesPermutation(p,i.shape.length),h=i;null!=c&&(h=J({inputs:{x:i},backend:n,attrs:{perm:c}}),p=r.backend_util.getInnerMostAxes(p.length,i.shape.length)),r.backend_util.assertAxesAreInnerMostDims("any",p,h.shape.length);let[d,f]=r.backend_util.computeOutAndReduceShapes(h.shape,p),m=r.util.sizeFromShape(f),g=r.util.makeZerosTypedArray(r.util.sizeFromShape(d),h.dtype),y=n.data.get(h.dataId).values;for(let b=0;b<g.length;++b){let k=b*m,N=y[k];for(let v=0;v<m;++v){let x=y[k+v];N=N||x}g[b]=N}null!=c&&n.disposeIntermediateTensorInfo(h);let w=n.makeTensorInfo(d,h.dtype,g);if(u){let T=r.backend_util.expandShapeToKeepDim(d,l),S=L({inputs:{x:w},backend:n,attrs:{shape:T}});return n.disposeIntermediateTensorInfo(w),S}return w}},er={kernelName:r.ArgMax,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o}=s;a(i,"argMax");let u=r.util.parseAxisParam(o,i.shape),l=r.backend_util.getAxesPermutation(u,i.shape.length),p=i,c=[];null!=l&&(c.push(p=J({inputs:{x:i},backend:n,attrs:{perm:l}})),u=r.backend_util.getInnerMostAxes(u.length,p.shape.length)),u=[u[0]],r.backend_util.assertAxesAreInnerMostDims("argMax",u,p.shape.length);let[h,d]=r.backend_util.computeOutAndReduceShapes(p.shape,u),f=r.util.sizeFromShape(h),m=r.util.makeZerosTypedArray(f,"int32"),g=r.util.sizeFromShape(d),y=n.data.get(p.dataId).values;for(let b=0;b<m.length;++b){let k=b*g,N=y[k],v=0;for(let x=0;x<g;++x){let w=y[k+x];w>N&&(N=w,v=x)}m[b]=v}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(h,"int32",m)}},ea={kernelName:r.ArgMin,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o}=s;a(i,"argMin");let u=r.util.parseAxisParam(o,i.shape),l=r.backend_util.getAxesPermutation(u,i.shape.length),p=i,c=[];null!=l&&(c.push(p=J({inputs:{x:i},backend:n,attrs:{perm:l}})),u=r.backend_util.getInnerMostAxes(u.length,p.shape.length)),u=[u[0]],r.backend_util.assertAxesAreInnerMostDims("argMin",u,p.shape.length);let[h,d]=r.backend_util.computeOutAndReduceShapes(p.shape,u),f=r.util.sizeFromShape(h),m=r.util.makeZerosTypedArray(f,"int32"),g=r.util.sizeFromShape(d),y=n.data.get(p.dataId).values;for(let b=0;b<m.length;++b){let k=b*g,N=y[k],v=0;for(let x=0;x<g;++x){let w=y[k+x];w<N&&(N=w,v=x)}m[b]=v}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(h,"int32",m)}},es=o(r.Asin,e=>Math.asin(e)),ei={kernelName:r.Asin,backendName:"cpu",kernelFunc:es},eo=o(r.Asinh,e=>Math.asinh(e)),eu={kernelName:r.Asinh,backendName:"cpu",kernelFunc:eo},el=o(r.Atan,e=>Math.atan(e)),ep={kernelName:r.Atan,backendName:"cpu",kernelFunc:el},ec=m((e,t)=>Math.atan2(e,t)),eh=B(r.Atan2,ec),ed={kernelName:r.Atan2,backendName:"cpu",kernelFunc:eh},ef=o(r.Atanh,e=>Math.atanh(e)),em={kernelName:r.Atanh,backendName:"cpu",kernelFunc:ef};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function eg(e,t,n,a,s,i){let o=s.strideHeight,u=s.strideWidth,l=s.dilationHeight,p=s.dilationWidth,c=s.effectiveFilterHeight,h=s.effectiveFilterWidth,d=s.padInfo.top,f=s.padInfo.left,m="max"===i?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,g=(0,r.buffer)(s.outShape,n),y=g.values,b=s.outShape[1]*s.outShape[2]*s.outShape[3],k=s.outShape[2]*s.outShape[3],N=s.outShape[3];for(let v=0;v<s.batchSize;++v){let x=v*b,w=v*a[0];for(let T=0;T<s.inChannels;++T)for(let S=0;S<s.outHeight;++S){let I=S*o-d,_=Math.max(0,I),E=Math.min(s.inHeight,c+I),A=x+S*k;for(let M=0;M<s.outWidth;++M){let D=M*u-f,$=Math.max(0,D),F=Math.min(s.inWidth,h+D),B=m,O=0,R=0;for(let C=_;C<E;C+=l){let V=w+C*a[1];for(let P=$;P<F;P+=p){let L=V+P*a[2],z=e[L+T];"max"===i&&z>B?B=z:"avg"===i&&(O+=z,R++)}if(isNaN(B))break}let W=A+M*N+T;y[W]="avg"===i?O/R:B}}}return g}function ey(e,t,n,a,s=!1,i=!1){let o=(0,r.buffer)(a.outShape,"int32"),u=a.strideHeight,l=a.strideWidth,p=a.dilationHeight,c=a.dilationWidth,h=a.effectiveFilterHeight,d=a.effectiveFilterWidth,f=a.padInfo.top,m=a.padInfo.left,g=(0,r.buffer)(t,n,e);for(let y=0;y<a.batchSize;++y)for(let b=0;b<a.inChannels;++b)for(let k=0;k<a.outHeight;++k){let N=k*u-f,v=N;for(;v<0;)v+=p;let x=Math.min(a.inHeight,h+N);for(let w=0;w<a.outWidth;++w){let T=w*l-m,S=T;for(;S<0;)S+=c;let I=Math.min(a.inWidth,d+T),_=Number.NEGATIVE_INFINITY,E=-1;for(let A=v;A<x;A+=p){let M=A-N;for(let D=S;D<I;D+=c){let $=D-T,F=g.get(y,A,D,b);F>_&&(_=F,E=s?i?((y*a.inHeight+A)*a.inWidth+D)*a.inChannels+b:(A*a.inWidth+D)*a.inChannels+b:M*d+$)}}o.set(E,y,k,w,b)}}return o}function eb(e,t,n,a,s,i){let o=s.strideDepth,u=s.strideHeight,l=s.strideWidth,p=s.dilationDepth,c=s.dilationHeight,h=s.dilationWidth,d=s.effectiveFilterDepth,f=s.effectiveFilterHeight,m=s.effectiveFilterWidth,g=s.padInfo.front,y=s.padInfo.top,b=s.padInfo.left,k="max"===i?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,N=(0,r.buffer)(s.outShape,n),v=N.values,x=s.outShape[1]*s.outShape[2]*s.outShape[3]*s.outShape[4],w=s.outShape[2]*s.outShape[3]*s.outShape[4],T=s.outShape[3]*s.outShape[4],S=s.outShape[4];for(let I=0;I<s.batchSize;++I){let _=I*x,E=I*a[0];for(let A=0;A<s.inChannels;++A)for(let M=0;M<s.outDepth;++M){let D=M*o-g,$=D;for(;$<0;)$+=p;let F=Math.min(s.inDepth,d+D),B=_+M*w;for(let O=0;O<s.outHeight;++O){let R=O*u-y,C=R;for(;C<0;)C+=c;let V=Math.min(s.inHeight,f+R),P=B+O*T;for(let L=0;L<s.outWidth;++L){let z=L*l-b,W=z;for(;W<0;)W+=h;let U=Math.min(s.inWidth,m+z),G=P+L*S,q=k,H=0,j=0;for(let K=$;K<F;K+=p){let X=E+K*a[1];for(let Z=C;Z<V;Z+=c){let Q=X+Z*a[2];for(let Y=W;Y<U;Y+=h){let J=Q+Y*a[3],ee=e[J+A];if("max"===i&&ee>q?q=ee:"avg"===i&&(H+=ee,j++),isNaN(q))break}if(isNaN(q))break}if(isNaN(q))break}let et=G+A;v[et]="avg"===i?H/j:q}}}}return N}let ek={kernelName:r.AvgPool,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t;a(i,"avgPool");let{filterSize:o,strides:u,pad:l,dimRoundingMode:p}=s;r.util.assert(r.backend_util.eitherStridesOrDilationsAreOne(u,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${u} and dilations '1'`);let h=r.backend_util.computePool2DInfo(i.shape,o,u,1,l,p),d;if(1===h.filterWidth&&1===h.filterHeight&&r.util.arraysEqual(h.inShape,h.outShape))d=c({inputs:{x:i},backend:n});else{let f=n.data.get(i.dataId).values,m=r.util.computeStrides(i.shape),g=eg(f,i.shape,i.dtype,m,h,"avg");d=n.makeTensorInfo(h.outShape,i.dtype,g.values)}return d}},eN={kernelName:r.AvgPool3D,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{filterSize:o,strides:u,pad:l,dimRoundingMode:p,dataFormat:c}=s;a(i,"avgPool3d");let h=r.backend_util.computePool3DInfo(i.shape,o,u,1,l,p,c),d=n.data.get(i.dataId).values,f=eb(d,i.shape,i.dtype,r.util.computeStrides(i.shape),h,"avg");return n.makeTensorInfo(f.shape,"float32",f.values)}},ev={kernelName:r.AvgPool3DGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{dy:i,input:o}=t,{filterSize:u,strides:l,pad:p,dimRoundingMode:c}=s;a([i,o],"avgPool3DGrad");let h=r.backend_util.computePool3DInfo(o.shape,u,l,1,p,c),d=h.strideDepth,f=h.strideHeight,m=h.strideWidth,g=h.filterDepth,y=h.filterHeight,b=h.filterWidth,k=h.dilationDepth,N=h.dilationHeight,v=h.dilationWidth,x=h.effectiveFilterDepth,w=h.effectiveFilterHeight,T=h.effectiveFilterWidth,S=x-1-h.padInfo.front,I=T-1-h.padInfo.left,_=w-1-h.padInfo.top,E=(0,r.buffer)(o.shape,"float32"),A=1/(g*y*b),M=n.bufferSync(i);for(let D=0;D<h.batchSize;++D)for(let $=0;$<h.inChannels;++$)for(let F=0;F<h.inDepth;++F)for(let B=0;B<h.inHeight;++B)for(let O=0;O<h.inWidth;++O){let R=F-S,C=B-_,V=O-I,P=0;for(let L=0;L<x;L+=k){let z=(R+L)/d;if(!(z<0)&&!(z>=h.outDepth)&&Math.floor(z)===z)for(let W=0;W<w;W+=N){let U=(C+W)/f;if(!(U<0)&&!(U>=h.outHeight)&&Math.floor(U)===U)for(let G=0;G<T;G+=v){let q=(V+G)/m;if(q<0||q>=h.outWidth||Math.floor(q)!==q)continue;let H=M.get(D,z,U,q,$);P+=H}}}E.set(P*A,D,F,B,O,$)}return n.makeTensorInfo(E.shape,E.dtype,E.values)}},ex={kernelName:r.AvgPoolGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{dy:i,input:o}=t;a([i,o],"avgPoolGrad");let{filterSize:u,strides:l,pad:p}=s,c=r.backend_util.computePool2DInfo(o.shape,u,l,1,p),h=c.strideHeight,d=c.strideWidth,f=c.filterHeight,m=c.filterWidth,g=c.dilationHeight,y=c.dilationWidth,b=c.effectiveFilterHeight,k=c.effectiveFilterWidth,N=k-1-c.padInfo.left,v=b-1-c.padInfo.top,x=(0,r.buffer)(o.shape,"float32"),w=1/(f*m),T=n.data.get(i.dataId).values,S=(0,r.buffer)(i.shape,"float32",T);for(let I=0;I<c.batchSize;++I)for(let _=0;_<c.inChannels;++_)for(let E=0;E<c.inHeight;++E)for(let A=0;A<c.inWidth;++A){let M=E-v,D=A-N,$=0;for(let F=0;F<b;F+=g){let B=(M+F)/h;if(!(B<0)&&!(B>=c.outHeight)&&Math.floor(B)===B)for(let O=0;O<k;O+=y){let R=(D+O)/d;if(R<0||R>=c.outWidth||Math.floor(R)!==R)continue;let C=S.get(I,B,R,_);$+=C}}x.set($*w,I,E,A,_)}return n.makeTensorInfo(x.shape,x.dtype,x.values)}},ew={kernelName:r.FusedBatchNorm,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,scale:o,offset:u,mean:l,variance:p}=t;r.util.assert(l.shape.length===p.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),r.util.assert(null==u||l.shape.length===u.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),r.util.assert(null==o||l.shape.length===o.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks."),a([i,l,p,o,u],"batchNorm");let{varianceEpsilon:c}=s;null==c&&(c=.001);let h=n.data.get(i.dataId).values,d=n.data.get(l.dataId).values,f=n.data.get(p.dataId).values,m=o?n.data.get(o.dataId).values:new Float32Array([1]),g=u?n.data.get(u.dataId).values:new Float32Array([0]),y=new Float32Array(h.length),b=g.length,k=m.length,N=f.length,v=d.length,x=0,w=0,T=0,S=0;for(let I=0;I<h.length;++I)y[I]=g[x++]+(h[I]-d[w++])*m[T++]/Math.sqrt(f[S++]+c),x>=b&&(x=0),w>=v&&(w=0),T>=k&&(T=0),S>=N&&(S=0);return n.makeTensorInfo(i.shape,i.dtype,y)}};function eT(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{begin:o,size:u}=s;a(i,"slice");let[l,p]=r.slice_util.parseSliceParams(i,o,u);r.slice_util.assertParamsValid(i,l,p);let c=n.data.get(i.dataId).values,h=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a,s){let i=r.slice_util.isSliceContinous(a,t,n),o=r.util.sizeFromShape(n),u=r.util.computeStrides(a);if(i){let l=r.slice_util.computeFlatOffset(t,u);return"string"===s?e.slice(l,l+o):e.subarray(l,l+o)}let p="string"===s?r.backend_util.fromUint8ToStringArray(e):e,c=(0,r.buffer)(a,s,p),h=(0,r.buffer)(n,s);for(let d=0;d<h.size;++d){let f=h.indexToLoc(d),m=f.map((e,n)=>e+t[n]);h.set(c.get(...m),...f)}return"string"===s?r.backend_util.fromStringArrayToUint8(h.values):h.values}(c,l,p,i.shape,i.dtype);return n.makeTensorInfo(p,i.dtype,h)}let eS={kernelName:r.Slice,backendName:"cpu",kernelFunc:eT},eI={kernelName:r.BatchToSpaceND,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{blockShape:o,crops:u}=s;a([i],"batchToSpaceND");let l=o.reduce((e,t)=>e*t),p=r.backend_util.getReshaped(i.shape,o,l),c=r.backend_util.getPermuted(p.length,o.length),h=r.backend_util.getReshapedPermuted(i.shape,o,l),d=r.backend_util.getSliceBeginCoords(u,o.length),f=r.backend_util.getSliceSize(h,u,o.length),m=L({inputs:{x:i},backend:n,attrs:{shape:p}}),g=J({inputs:{x:m},backend:n,attrs:{perm:c}}),y=L({inputs:{x:g},backend:n,attrs:{shape:h}}),b=eT({inputs:{x:y},backend:n,attrs:{begin:d,size:f}});return n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(y),b}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function e_(e,t,n,a,s){let i=r.util.sizeFromShape(a),o=r.util.makeZerosTypedArray(s,n);for(let u=0;u<e.length;u++){let l=e[u];if(l<0)throw Error("Input x must be non-negative!");!(l>=s)&&(i>0?o[l]+=t[u]:o[l]+=1)}return o}let eE={kernelName:r.Bincount,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:r}=e,{x:a,weights:s}=t,{size:i}=r,o=n.data.get(a.dataId).values,u=n.data.get(s.dataId).values,l=e_(o,u,s.dtype,s.shape,i);return n.makeTensorInfo([i],s.dtype,l)}},eA={kernelName:r.BroadcastArgs,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{s0:a,s1:s}=t,i=n.data.get(a.dataId).values,o=n.data.get(s.dataId).values,u=r.backend_util.assertAndGetBroadcastShape(Array.from(i),Array.from(o));return n.makeTensorInfo([u.length],"int32",Int32Array.from(u))}},eM=w(e=>Math.ceil(e)),eD=u(r.Ceil,eM),e$={kernelName:r.Ceil,backendName:"cpu",kernelFunc:eD},eF=o(r.ClipByValue,(e,t)=>e>t.clipValueMax?t.clipValueMax:e<t.clipValueMin?t.clipValueMin:e),eB={kernelName:r.ClipByValue,backendName:"cpu",kernelFunc:eF},eO=e=>{let{x:t}=e.inputs,n=e.backend,a=new Float32Array(r.util.sizeFromShape(t.shape)),s=n.data.get(t.dataId),i=s.complexTensorInfos.real,o=s.complexTensorInfos.imag,u=n.data.get(i.dataId).values,l=n.data.get(o.dataId).values;for(let p=0;p<u.length;p++){let c=u[p],h=l[p];a[p]=Math.hypot(c,h)}return n.makeOutput(a,t.shape,"float32")},eR={kernelName:r.ComplexAbs,backendName:"cpu",kernelFunc:eO};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function eC(e){let{inputs:t,backend:n}=e,{input:r}=t,a=n.data.get(r.dataId).complexTensorInfos.imag,s=n.data.get(a.dataId).values;return n.makeTensorInfo(a.shape,a.dtype,s)}let eV={kernelName:r.Imag,backendName:"cpu",kernelFunc:eC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function eP(e){let{inputs:t,backend:n,attrs:a}=e,{axis:s}=a,i=r.util.parseAxisParam(s,t[0].shape)[0],o=t.map(e=>e.shape);r.backend_util.assertParamsConsistent(o,i);let u=r.backend_util.computeOutShape(t.map(e=>e.shape),i);if(0===r.util.sizeFromShape(u))return n.makeTensorInfo(u,t[0].dtype,[]);let l=t.filter(e=>r.util.sizeFromShape(e.shape)>0);if(1===l.length)return c({inputs:{x:l[0]},backend:n});if("complex64"===l[0].dtype){let p=l.map(e=>M({inputs:{input:e},backend:n})),h=l.map(e=>eC({inputs:{input:e},backend:n})),d=eP({inputs:p,backend:n,attrs:{axis:i}}),f=eP({inputs:h,backend:n,attrs:{axis:i}}),m=_({inputs:{real:d,imag:f},backend:n});return p.forEach(e=>n.disposeIntermediateTensorInfo(e)),h.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),m}let g=l.map(e=>{let t=r.util.sizeFromShape(e.shape.slice(i)),a=[-1,t];return L({inputs:{x:e},backend:n,attrs:{shape:a}})}),y=g.map(e=>({vals:n.data.get(e.dataId).values,shape:e.shape}));u=r.backend_util.computeOutShape(g.map(e=>e.shape),1);let b=1===g[0].shape[0],k=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a){let s=r.util.getArrayFromDType(n,r.util.sizeFromShape(t));if(a&&"string"!==n){let i=0;e.forEach(e=>{let t=r.util.sizeFromShape(e.shape);s.set(e.vals,i),i+=t})}else{let o=0;e.forEach(e=>{let a="string"===n?r.backend_util.fromUint8ToStringArray(e.vals):e.vals,i=0;for(let u=0;u<e.shape[0];++u){let l=u*t[1]+o;for(let p=0;p<e.shape[1];++p)s[l+p]=a[i++]}o+=e.shape[1]})}return s}(y,u,t[0].dtype,b),N=r.backend_util.computeOutShape(l.map(e=>e.shape),i),v=n.makeTensorInfo(N,t[0].dtype,k);return g.forEach(e=>n.disposeIntermediateTensorInfo(e)),v}let eL={kernelName:r.Concat,backendName:"cpu",kernelFunc:eP};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function ez(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,filter:o}=t,{strides:u,pad:l,dataFormat:p,dilations:c,dimRoundingMode:h}=s;a([i,o],"conv2d");let d=r.backend_util.convertConv2DDataFormat(p),f=r.backend_util.computeConv2DInfo(i.shape,o.shape,u,c,l,h,!1,d),m=f.filterHeight,g=f.filterWidth,y=f.dilationHeight,b=f.dilationWidth,k=f.padInfo.left,N=f.padInfo.top,v="channelsLast"===f.dataFormat,x=new r.TensorBuffer(f.outShape,i.dtype),w=r.util.computeStrides(i.shape),T=r.util.computeStrides(o.shape),S=w[0],I=v?w[1]:w[2],_=v?w[2]:1,E=v?1:w[1],A=x.strides[0],M=v?x.strides[1]:x.strides[2],D=v?x.strides[2]:1,$=v?1:x.strides[1],F=n.data.get(i.dataId).values,B=n.data.get(o.dataId).values,O=x.values;for(let R=0;R<f.batchSize;++R){let C=R*S,V=R*A;for(let P=0;P<f.outHeight;++P){let L=V+P*M,z=P*f.strideHeight-N;for(let W=0;W<m;++W){let U=z+W*y;if(U<0||U>=f.inHeight)continue;let G=W*T[0],q=C+U*I;for(let H=0;H<f.outWidth;++H){let j=L+H*D,K=H*f.strideWidth-k;for(let X=0;X<g;++X){let Z=K+X*b;if(Z<0||Z>=f.inWidth)continue;let Q=G+X*T[1],Y=q+Z*_,J=Q;for(let ee=0;ee<f.inChannels;++ee){let et=F[Y+ee*E];for(let en=0;en<f.outChannels;++en)O[j+en*$]+=et*B[J+en];J+=f.outChannels}}}}}}return n.makeTensorInfo(x.shape,x.dtype,O)}let eW={kernelName:r.Conv2D,backendName:"cpu",kernelFunc:ez},eU={kernelName:r.Conv2DBackpropFilter,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,dy:o}=t,{strides:u,pad:l,dataFormat:p,dimRoundingMode:c,filterShape:h}=s;a([i,o],"conv2dBackpropFilter");let d=r.backend_util.convertConv2DDataFormat(p),f=r.backend_util.computeConv2DInfo(i.shape,h,u,1,l,c,!1,d),{strideHeight:m,strideWidth:g,filterHeight:y,filterWidth:b}=f,k="channelsLast"===f.dataFormat,N=new r.TensorBuffer(f.filterShape,"float32"),v=f.padInfo.left,x=f.padInfo.top,w=n.data.get(i.dataId).values,T=n.data.get(o.dataId).values,S=new r.TensorBuffer(i.shape,i.dtype,w),I=new r.TensorBuffer(o.shape,o.dtype,T);for(let _=0;_<y;++_){let E=Math.max(0,Math.ceil((x-_)/m)),A=Math.min(f.outHeight,(f.inHeight+x-_)/m);for(let M=0;M<b;++M){let D=Math.max(0,Math.ceil((v-M)/g)),$=Math.min(f.outWidth,(f.inWidth+v-M)/g);for(let F=0;F<f.inChannels;++F)for(let B=0;B<f.outChannels;++B){let O=0;for(let R=0;R<f.batchSize;++R)for(let C=E;C<A;++C){let V=_+C*m-x;for(let P=D;P<$;++P){let L=M+P*g-v;k?O+=S.get(R,V,L,F)*I.get(R,C,P,B):O+=S.get(R,F,V,L)*I.get(R,B,C,P)}}N.set(O,_,M,F,B)}}}return n.makeTensorInfo(N.shape,N.dtype,N.values)}},eG={kernelName:r.Conv2DBackpropInput,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{dy:i,filter:o}=t,{inputShape:u,strides:l,pad:p,dataFormat:c,dimRoundingMode:h}=s;a([i,o],"conv2dBackpropInput");let d=r.util.computeStrides(o.shape),f=r.util.computeStrides(i.shape),m=r.backend_util.convertConv2DDataFormat(c),g=r.backend_util.computeConv2DInfo(u,o.shape,l,1,p,h,!1,m),y=new r.TensorBuffer(g.inShape,"float32"),b=y.values,k=n.data.get(i.dataId).values,N=n.data.get(o.dataId).values,[v,x,w]=d,{batchSize:T,filterHeight:S,filterWidth:I,inChannels:_,inHeight:E,inWidth:A,outChannels:M,outHeight:D,outWidth:$,strideHeight:F,strideWidth:B}=g;m=g.dataFormat;let O=S-1-g.padInfo.top,R=I-1-g.padInfo.left,C="channelsLast"===m,V=y.strides[0],P=C?y.strides[1]:y.strides[2],L=C?y.strides[2]:1,z=C?1:y.strides[1],W=f[0],U=C?f[1]:f[2],G=C?f[2]:1,q=C?1:f[1];for(let H=0;H<T;++H)for(let j=0;j<_;++j)for(let K=0;K<E;++K){let X=K-O,Z=Math.max(0,Math.ceil(X/F)),Q=Math.min(D,(S+X)/F);for(let Y=0;Y<A;++Y){let J=Y-R,ee=Math.max(0,Math.ceil(J/B)),et=Math.min($,(I+J)/B),en=0;for(let er=Z;er<Q;++er){let ea=er*F-X;for(let es=ee;es<et;++es){let ei=es*B-J,eo=W*H+U*er+G*es,eu=v*(S-1-ea)+x*(I-1-ei)+w*j;for(let el=0;el<M;++el){let ep=k[eo+q*el],ec=N[eu+el];en+=ep*ec}}}let eh=V*H+P*K+L*Y+z*j;b[eh]=en}}return n.makeTensorInfo(y.shape,y.dtype,y.values)}},eq={kernelName:r.Conv3D,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,filter:o}=t,{strides:u,pad:l,dilations:p}=s;a([i,o],"conv3d");let c=r.backend_util.computeConv3DInfo(i.shape,o.shape,u,p,l),{filterDepth:h,filterHeight:d,filterWidth:f,dilationDepth:m,dilationHeight:g,dilationWidth:y,padInfo:b}=c,k=b.front,N=b.left,v=b.top,x=new r.TensorBuffer(c.outShape,i.dtype),w=n.data.get(i.dataId).values,T=n.data.get(o.dataId).values,S=x.values,I=r.util.computeStrides(i.shape),_=r.util.computeStrides(o.shape);for(let E=0;E<c.batchSize;++E){let A=E*I[0],M=E*x.strides[0];for(let D=0;D<c.outDepth;++D){let $=M+D*x.strides[1],F=D*c.strideDepth-k;for(let B=0;B<h;++B){let O=F+B*m;if(O<0||O>=c.inDepth)continue;let R=B*_[0],C=A+O*I[1];for(let V=0;V<c.outHeight;++V){let P=$+V*x.strides[2],L=V*c.strideHeight-v;for(let z=0;z<d;++z){let W=L+z*g;if(W<0||W>=c.inHeight)continue;let U=R+z*_[1],G=C+W*I[2];for(let q=0;q<c.outWidth;++q){let H=P+q*c.outChannels,j=q*c.strideWidth-N;for(let K=0;K<f;++K){let X=j+K*y;if(X<0||X>=c.inWidth)continue;let Z=U+K*_[2],Q=G+X*c.inChannels,Y=Z;for(let J=0;J<c.inChannels;++J){let ee=w[Q+J];for(let et=0;et<c.outChannels;++et)S[H+et]+=ee*T[Y+et];Y+=c.outChannels}}}}}}}}return n.makeTensorInfo(x.shape,x.dtype,x.values)}},eH={kernelName:r.Conv3DBackpropFilterV2,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,dy:o}=t,{strides:u,pad:l,filterShape:p}=s;a([i,o],"conv3dBackpropFilterV2");let c=r.util.computeStrides(i.shape),h=r.util.computeStrides(o.shape),d=r.backend_util.computeConv3DInfo(i.shape,p,u,1,l),f=d.strideDepth,m=d.strideHeight,g=d.strideWidth,y=d.filterDepth,b=d.filterHeight,k=d.filterWidth,N=new r.TensorBuffer(d.filterShape,"float32"),v=N.values,[x,w,T,S]=N.strides,I=n.data.get(o.dataId).values,[_,E,A,M]=h,D=n.data.get(i.dataId).values,[$,F,B,O]=c,R=d.padInfo.front,C=d.padInfo.left,V=d.padInfo.top;for(let P=0;P<y;++P){let L=Math.max(0,Math.ceil((R-P)/f)),z=Math.min(d.outDepth,(d.inDepth+R-P)/f),W=P*x;for(let U=0;U<b;++U){let G=Math.max(0,Math.ceil((V-U)/m)),q=Math.min(d.outHeight,(d.inHeight+V-U)/m),H=U*w+W;for(let j=0;j<k;++j){let K=Math.max(0,Math.ceil((C-j)/g)),X=Math.min(d.outWidth,(d.inWidth+C-j)/g),Z=j*T+H;for(let Q=0;Q<d.inChannels;++Q){let Y=Q*S+Z;for(let J=0;J<d.outChannels;++J){let ee=0;for(let et=0;et<d.batchSize;++et){let en=et*$,er=et*_;for(let ea=L;ea<z;++ea){let es=P+ea*f-R,ei=es*F+en,eo=ea*E+er;for(let eu=G;eu<q;++eu){let el=U+eu*m-V,ep=el*B+ei,ec=eu*A+eo;for(let eh=K;eh<X;++eh){let ed=j+eh*g-C,ef=ed*O+ep,em=eh*M+ec;ee+=D[ef+Q]*I[em+J]}}}}v[Y+J]=ee}}}}}return n.makeTensorInfo(N.shape,N.dtype,N.values)}},ej={kernelName:r.Conv3DBackpropInputV2,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{dy:i,filter:o}=t,{pad:u,strides:l,inputShape:p}=s;a([i],"conv3dBackpropInputV2");let c=r.util.computeStrides(i.shape),h=r.util.computeStrides(o.shape),d=r.backend_util.computeConv3DInfo(p,o.shape,l,1,u),f=new r.TensorBuffer(d.inShape,"float32"),m=f.values,[g,y,b,k]=f.strides,N=n.data.get(i.dataId).values,[v,x,w,T]=c,S=n.data.get(o.dataId).values,[I,_,E,A]=h,{batchSize:M,filterDepth:D,filterHeight:$,filterWidth:F,inChannels:B,inDepth:O,inHeight:R,inWidth:C,outChannels:V,outDepth:P,outHeight:L,outWidth:z,strideDepth:W,strideHeight:U,strideWidth:G}=d,q=D-1-d.padInfo.front,H=$-1-d.padInfo.top,j=F-1-d.padInfo.left;for(let K=0;K<M;++K)for(let X=0;X<B;++X)for(let Z=0;Z<O;++Z){let Q=Z-q,Y=Math.max(0,Math.ceil(Q/W)),J=Math.min(P,(D+Q)/W);for(let ee=0;ee<R;++ee){let et=ee-H,en=Math.max(0,Math.ceil(et/U)),er=Math.min(L,($+et)/U);for(let ea=0;ea<C;++ea){let es=ea-j,ei=Math.max(0,Math.ceil(es/G)),eo=Math.min(z,(F+es)/G),eu=0;for(let el=Y;el<J;++el){let ep=el*W-Q;for(let ec=en;ec<er;++ec){let eh=ec*U-et;for(let ed=ei;ed<eo;++ed){let ef=ed*G-es,em=v*K+x*el+w*ec+T*ed,eg=I*(D-1-ep)+_*($-1-eh)+E*(F-1-ef)+A*X;for(let ey=0;ey<V;++ey){let eb=N[em+ey],ek=S[eg+ey];eu+=eb*ek}}}}m[g*K+y*Z+b*ee+k*ea+X]=eu}}}return n.makeTensorInfo(f.shape,f.dtype,f.values)}},eK=o(r.Cos,e=>Math.cos(e)),eX={kernelName:r.Cos,backendName:"cpu",kernelFunc:eK},eZ=o(r.Cosh,e=>Math.cosh(e)),eQ={kernelName:r.Cosh,backendName:"cpu",kernelFunc:eZ},eY={kernelName:r.CropAndResize,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{image:s,boxes:i,boxInd:o}=t,{cropSize:u,method:l,extrapolationValue:p}=a,[c,h,d,f]=s.shape,m=i.shape[0],[g,y]=u,b=(0,r.buffer)([m,g,y,f],"float32"),k=n.data.get(i.dataId).values,N=n.data.get(o.dataId).values,v=n.data.get(s.dataId).values,x=r.util.computeStrides(s.shape),w=r.util.computeStrides(b.shape);for(let T=0;T<m;T++){let S=4*T,I=k[S],_=k[S+1],E=k[S+2],A=k[S+3],M=N[T];if(M>=c)continue;let D=g>1?(E-I)*(h-1)/(g-1):0,$=y>1?(A-_)*(d-1)/(y-1):0;for(let F=0;F<g;F++){let B=g>1?I*(h-1)+F*D:.5*(I+E)*(h-1);if(B<0||B>h-1){for(let O=0;O<y;O++)for(let R=0;R<f;R++){let C=R+O*w[2]+F*w[1]+T*w[0];b.values[C]=p}continue}if("bilinear"===l){let V=Math.floor(B),P=Math.ceil(B),L=B-V;for(let z=0;z<y;z++){let W=y>1?_*(d-1)+z*$:.5*(_+A)*(d-1);if(W<0||W>d-1){for(let U=0;U<f;U++){let G=U+z*w[2]+F*w[1]+T*w[0];b.values[G]=p}continue}let q=Math.floor(W),H=Math.ceil(W),j=W-q;for(let K=0;K<f;K++){let X=K+q*x[2]+V*x[1]+M*x[0],Z=v[X];X=K+H*x[2]+V*x[1]+M*x[0];let Q=v[X];X=K+q*x[2]+P*x[1]+M*x[0];let Y=v[X];X=K+H*x[2]+P*x[1]+M*x[0];let J=v[X],ee=Z+(Q-Z)*j,et=Y+(J-Y)*j;X=K+z*w[2]+F*w[1]+T*w[0],b.values[X]=ee+(et-ee)*L}}}else for(let en=0;en<y;++en){let er=y>1?_*(d-1)+en*$:.5*(_+A)*(d-1);if(er<0||er>d-1){for(let ea=0;ea<f;ea++){let es=ea+en*w[2]+F*w[1]+T*w[0];b.values[es]=p}continue}let ei=Math.round(er),eo=Math.round(B);for(let eu=0;eu<f;eu++){let el=eu+ei*x[2]+eo*x[1]+M*x[0],ep=eu+en*w[2]+F*w[1]+T*w[0];b.values[ep]=v[el]}}}}return n.makeTensorInfo(b.shape,b.dtype,b.values)}},eJ={kernelName:r.Cumprod,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o,exclusive:u,reverse:l}=s;a(i,"cumprod");let p=r.backend_util.getAxesPermutation([o],i.shape.length),c=i;null!=p&&(c=J({inputs:{x:i},backend:n,attrs:{perm:p}}));let h=r.backend_util.getInnerMostAxes(1,i.shape.length)[0];if(h!==c.shape.length-1)throw Error(`backend.cumprod in CPU expects an inner-most axis=${c.shape.length-1} but got axis=${h}`);let d=(0,r.upcastType)(c.dtype,"int32"),f=r.util.makeOnesTypedArray(r.util.sizeFromShape(c.shape),d),m=n.data.get(c.dataId).values,g=c.shape[c.shape.length-1],y=l?(e,t)=>e+g-t-1:(e,t)=>e+t;for(let b=0;b<m.length;b+=g)for(let k=0;k<g;k++){let N=y(b,k);if(0===k)f[N]=u?1:m[N];else{let v=y(b,k-1);f[N]=u?m[v]*f[v]:m[N]*f[v]}}let x=n.makeTensorInfo(c.shape,d,f);if(null!=p){let w=r.backend_util.getUndoAxesPermutation(p),T=J({inputs:{x:x},backend:n,attrs:{perm:w}});return n.disposeIntermediateTensorInfo(x),n.disposeIntermediateTensorInfo(c),T}return x}},e0={kernelName:r.Cumsum,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o,exclusive:u,reverse:l}=s;a(i,"cumsum");let p=r.backend_util.getAxesPermutation([o],i.shape.length),c=i;null!=p&&(c=J({inputs:{x:i},backend:n,attrs:{perm:p}}));let h=r.backend_util.getInnerMostAxes(1,i.shape.length)[0];if(h!==c.shape.length-1)throw Error(`backend.cumsum in CPU expects an inner-most axis=${c.shape.length-1} but got axis=${h}`);let d=(0,r.upcastType)(c.dtype,"int32"),f=r.util.makeZerosTypedArray(r.util.sizeFromShape(c.shape),d),m=n.data.get(c.dataId).values,g=c.shape[c.shape.length-1],y=l?(e,t)=>e+g-t-1:(e,t)=>e+t;for(let b=0;b<m.length;b+=g)for(let k=0;k<g;k++){let N=y(b,k);if(0===k)f[N]=u?0:m[N];else{let v=y(b,k-1);f[N]=u?m[v]+f[v]:m[N]+f[v]}}let x=n.makeTensorInfo(c.shape,d,f);if(null!=p){let w=r.backend_util.getUndoAxesPermutation(p),T=J({inputs:{x:x},backend:n,attrs:{perm:w}});return n.disposeIntermediateTensorInfo(x),n.disposeIntermediateTensorInfo(c),T}return x}},e1={kernelName:r.DenseBincount,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{x:s,weights:i}=t,{size:o,binaryOutput:u}=a;if(1===s.shape.length){let l=n.data.get(s.dataId).values,p=n.data.get(i.dataId).values,c=e_(l,p,i.dtype,i.shape,o);return n.makeTensorInfo([o],i.dtype,c)}if(2===s.shape.length){let h=n.bufferSync(s),d=n.bufferSync(i),f=function(e,t,n,a=!1){let s=e.shape[0],i=e.shape[1],o=(0,r.buffer)([s,n],t.dtype);for(let u=0;u<s;u++)for(let l=0;l<i;l++){let p=e.get(u,l);if(p<0)throw Error("Input x must be non-negative!");!(p>=n)&&(a?o.set(1,u,p):t.size>0?o.set(o.get(u,p)+t.get(u,l),u,p):o.set(o.get(u,p)+1,u,p))}return o}(h,d,o,u);return n.makeTensorInfo(f.shape,i.dtype,f.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${s.shape.length}.`)}},e2={kernelName:r.DepthToSpace,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{x:s}=t,{blockSize:i,dataFormat:o}=a;r.util.assert("NHWC"===o,()=>`Only NHWC dataFormat supported on CPU for depthToSpace. Got ${o}`);let u=s.shape[0],l=s.shape[1],p=s.shape[2],c=s.shape[3],h=l*i,d=p*i,f=c/(i*i),m=n.data.get(s.dataId).values,g=new Float32Array(u*h*d*f),y=0;for(let b=0;b<u;++b)for(let k=0;k<h;++k){let N=Math.floor(k/i),v=k%i;for(let x=0;x<d;++x){let w=Math.floor(x/i),T=x%i,S=(v*i+T)*f;for(let I=0;I<f;++I){let _=I+S,E=_+c*(w+p*(N+l*b));g[y++]=m[E]}}}return n.makeTensorInfo([u,h,d,f],s.dtype,g)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function e3(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,filter:o}=t,{strides:u,pad:l,dilations:p,dimRoundingMode:c}=s;a([i,o],"depthwiseConv2DNative");let h=r.util.computeStrides(i.shape),d=r.util.computeStrides(o.shape),f=p;null==f&&(f=[1,1]),r.util.assert(r.backend_util.eitherStridesOrDilationsAreOne(u,f),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${u} and dilations '${f}'`);let m=r.backend_util.computeConv2DInfo(i.shape,o.shape,u,f,l,c,!0),{filterHeight:g,filterWidth:y,dilationHeight:b,dilationWidth:k,padInfo:N}=m,v=N.left,x=N.top,w=m.outChannels/m.inChannels,T=new r.TensorBuffer(m.outShape,i.dtype),S=n.data.get(i.dataId).values,I=n.data.get(o.dataId).values,_=T.values;for(let E=0;E<m.batchSize;++E){let A=E*h[0],M=E*T.strides[0];for(let D=0;D<m.outHeight;++D){let $=M+D*T.strides[1],F=D*m.strideHeight-x;for(let B=0;B<g;++B){let O=F+B*b;if(O<0||O>=m.inHeight)continue;let R=B*d[0],C=A+O*h[1];for(let V=0;V<m.outWidth;++V){let P=$+V*T.strides[2],L=V*m.strideWidth-v;for(let z=0;z<y;++z){let W=L+z*k;if(W<0||W>=m.inWidth)continue;let U=R+z*d[1],G=C+W*m.inChannels,q=P,H=U;for(let j=0;j<m.inChannels;++j){let K=S[G+j];for(let X=0;X<w;++X)_[q+X]+=K*I[H+X];q+=w,H+=w}}}}}}return n.makeTensorInfo(T.shape,T.dtype,T.values)}let e6={kernelName:r.DepthwiseConv2dNative,backendName:"cpu",kernelFunc:e3},e4={kernelName:r.DepthwiseConv2dNativeBackpropFilter,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,dy:o}=t,{strides:u,dilations:l,pad:p,dimRoundingMode:c,filterShape:h}=s;a([i,o],"depthwiseConv2dNativeBackpropFilter");let d=r.backend_util.computeConv2DInfo(i.shape,h,u,l,p,c,!0),{strideHeight:f,strideWidth:m,filterHeight:g,filterWidth:y}=d,b=new r.TensorBuffer(d.filterShape,"float32"),k=d.padInfo.left,N=d.padInfo.top,v=d.outChannels/d.inChannels,x=n.data.get(i.dataId).values,w=new r.TensorBuffer(i.shape,i.dtype,x),T=n.data.get(o.dataId).values,S=new r.TensorBuffer(o.shape,o.dtype,T);for(let I=0;I<g;++I){let _=Math.max(0,Math.ceil((N-I)/f)),E=Math.min(d.outHeight,(d.inHeight+N-I)/f);for(let A=0;A<y;++A){let M=Math.max(0,Math.ceil((k-A)/m)),D=Math.min(d.outWidth,(d.inWidth+k-A)/m);for(let $=0;$<d.outChannels;++$){let F=Math.trunc($/v),B=$%v,O=0;for(let R=0;R<d.batchSize;++R)for(let C=_;C<E;++C){let V=I+C*f-N;for(let P=M;P<D;++P){let L=A+P*m-k;O+=w.get(R,V,L,F)*S.get(R,C,P,$)}}b.set(O,I,A,F,B)}}}return n.makeTensorInfo(b.shape,b.dtype,b.values)}},e5={kernelName:r.DepthwiseConv2dNativeBackpropInput,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{dy:i,filter:o}=t,{strides:u,dilations:l,pad:p,dimRoundingMode:c,inputShape:h}=s;a([i,o],"depthwiseConv2DNativeBackpropInput");let d=r.util.computeStrides(i.shape),f=r.util.computeStrides(o.shape),m=r.backend_util.computeConv2DInfo(h,o.shape,u,l,p,c,!0),g=new r.TensorBuffer(m.inShape,"float32"),y=g.values,[b,k,N]=g.strides,v=n.data.get(i.dataId).values,[x,w,T]=d,S=n.data.get(o.dataId).values,[I,_,E]=f,{batchSize:A,filterHeight:M,filterWidth:D,inChannels:$,inHeight:F,inWidth:B,outChannels:O,outHeight:R,outWidth:C,strideHeight:V,strideWidth:P}=m,L=M-1-m.padInfo.top,z=D-1-m.padInfo.left,W=O/$;for(let U=0;U<A;++U)for(let G=0;G<$;++G)for(let q=0;q<F;++q){let H=q-L,j=Math.max(0,Math.ceil(H/V)),K=Math.min(R,(M+H)/V);for(let X=0;X<B;++X){let Z=X-z,Q=Math.max(0,Math.ceil(Z/P)),Y=Math.min(C,(D+Z)/P),J=0;for(let ee=j;ee<K;++ee){let et=ee*V-H;for(let en=Q;en<Y;++en){let er=en*P-Z,ea=x*U+w*ee+T*en,es=I*(M-1-et)+_*(D-1-er)+E*G;for(let ei=0;ei<W;++ei){let eo=G*W+ei,eu=v[ea+eo],el=S[es+ei];J+=eu*el}}}y[b*U+k*q+N*X+G]=J}}return n.makeTensorInfo(g.shape,g.dtype,g.values)}},e8={kernelName:r.Diag,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{x:a}=t,s=r.util.sizeFromShape(a.shape),i=n.data.get(a.dataId).values,o=(0,r.buffer)([s,s],a.dtype),u=o.values;for(let l=0;l<i.length;l++)u[l*s+l]=i[l];let p=[...a.shape,...a.shape];return n.makeTensorInfo(p,o.dtype,o.values)}},e7={kernelName:r.Dilation2D,backendName:"cpu",kernelFunc({inputs:e,backend:t,attrs:n}){let{x:a,filter:s}=e,{strides:i,pad:o,dilations:u}=n,l=t.data.get(a.dataId).values,p=a.shape.length,c=t.data.get(s.dataId).values,h=s.shape.length,{batchSize:d,inHeight:f,inWidth:m,inChannels:g,outHeight:y,outWidth:b,padInfo:k,strideHeight:N,strideWidth:v,filterHeight:x,filterWidth:w,dilationHeight:T,dilationWidth:S,outShape:I}=r.backend_util.computeDilation2DInfo(a.shape,s.shape,i,o,"NHWC",u),_=r.util.sizeFromShape(I),E=I.length,A=r.util.getArrayFromDType(a.dtype,_);for(let M=0;M<d;++M)for(let D=0;D<y;++D){let $=D*N-k.top;for(let F=0;F<b;++F){let B=F*v-k.left;for(let O=0;O<g;++O){let R=Number.MIN_SAFE_INTEGER;for(let C=0;C<x;++C){let V=$+C*T;if(V>=0&&V<f)for(let P=0;P<w;++P){let L=B+P*S;if(L>=0&&L<m){let z=r.util.locToIndex([M,V,L,O],p,r.util.computeStrides(a.shape)),W=r.util.locToIndex([C,P,O],h,r.util.computeStrides(s.shape)),U=l[z]+c[W];U>R&&(R=U)}}}let G=r.util.locToIndex([M,D,F,O],E,r.util.computeStrides(I));A[G]=R}}}let q=t.write(r.util.toTypedArray(A,a.dtype),I,a.dtype);return{dataId:q,shape:I,dtype:a.dtype}}},e9={kernelName:r.Dilation2DBackpropFilter,backendName:"cpu",kernelFunc({inputs:e,backend:t,attrs:n}){let{x:a,filter:s,dy:i}=e,{strides:o,pad:u,dilations:l}=n,p=r.util.toNestedArray(a.shape,t.data.get(a.dataId).values),c=r.util.toNestedArray(s.shape,t.data.get(s.dataId).values),{batchSize:h,inHeight:d,inWidth:f,inChannels:m,outHeight:g,outWidth:y,padInfo:b,strideHeight:k,strideWidth:N,filterHeight:v,filterWidth:x,dilationHeight:w,dilationWidth:T,outShape:S}=r.backend_util.computeDilation2DInfo(a.shape,s.shape,o,u,"NHWC",l);r.util.assert(i.rank===S.length,()=>`Error in ${r.Dilation2DBackpropFilter}, dy must have the same rank as output ${S.length}, but got ${i.rank}`);let I=r.util.toNestedArray(S,t.data.get(i.dataId).values),_=r.util.makeZerosNestedTypedArray(s.shape,s.dtype);for(let E=0;E<h;++E)for(let A=0;A<g;++A){let M=A*k-b.top;for(let D=0;D<y;++D){let $=D*N-b.left;for(let F=0;F<m;++F){let B=Number.MIN_SAFE_INTEGER,O=0,R=0;for(let C=0;C<v;++C){let V=M+C*w;if(V>=0&&V<d)for(let P=0;P<x;++P){let L=$+P*T;if(L>=0&&L<f){let z=p[E][V][L][F]+c[C][P][F];z>B&&(B=z,O=C,R=P)}}}_[O][R][F]+=I[E][A][D][F]}}}let W=t.write(r.util.toTypedArray(_,a.dtype),s.shape,s.dtype);return{dataId:W,shape:s.shape,dtype:s.dtype}}},te={kernelName:r.Dilation2DBackpropInput,backendName:"cpu",kernelFunc({inputs:e,backend:t,attrs:n}){let{x:a,filter:s,dy:i}=e,{strides:o,pad:u,dilations:l}=n,p=r.util.toNestedArray(a.shape,t.data.get(a.dataId).values),c=r.util.toNestedArray(s.shape,t.data.get(s.dataId).values),{batchSize:h,inHeight:d,inWidth:f,inChannels:m,outHeight:g,outWidth:y,padInfo:b,strideHeight:k,strideWidth:N,filterHeight:v,filterWidth:x,dilationHeight:w,dilationWidth:T,outShape:S}=r.backend_util.computeDilation2DInfo(a.shape,s.shape,o,u,"NHWC",l);r.util.assert(i.rank===S.length,()=>`Error in ${r.Dilation2DBackpropInput}, dy must have the same rank as output ${S.length}, but got ${i.rank}`);let I=r.util.toNestedArray(S,t.data.get(i.dataId).values),_=r.util.makeZerosNestedTypedArray(a.shape,a.dtype);for(let E=0;E<h;++E)for(let A=0;A<g;++A){let M=A*k-b.top;for(let D=0;D<y;++D){let $=D*N-b.left;for(let F=0;F<m;++F){let B=Number.MIN_SAFE_INTEGER,O=M<0?0:M,R=$<0?0:$;for(let C=0;C<v;++C){let V=M+C*w;if(V>=0&&V<d)for(let P=0;P<x;++P){let L=$+P*T;if(L>=0&&L<f){let z=p[E][V][L][F]+c[C][P][F];z>B&&(B=z,O=V,R=L)}}}_[E][O][R][F]+=I[E][A][D][F]}}}let W=t.write(r.util.toTypedArray(_,a.dtype),a.shape,a.dtype);return{dataId:W,shape:a.shape,dtype:a.dtype}}},tt=m((e,t)=>e*t),tn=O((e,t,n,r)=>({real:e*n-t*r,imag:e*r+t*n})),tr=B(r.Multiply,tt,tn),ta={kernelName:r.Multiply,backendName:"cpu",kernelFunc:tr};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function ts(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o,keepDims:u}=s;a(i,"sum");let l;l="bool"===i.dtype?$({inputs:{x:i},backend:n,attrs:{dtype:"int32"}}):c({inputs:{x:i},backend:n});let p=l.shape.length,h=r.util.parseAxisParam(o,l.shape),d=r.backend_util.getAxesPermutation(h,p),f=h,m=l;null!=d&&(m=J({inputs:{x:l},backend:n,attrs:{perm:d}}),f=r.backend_util.getInnerMostAxes(f.length,p)),r.backend_util.assertAxesAreInnerMostDims("sum",f,m.shape.length);let[g,y]=r.backend_util.computeOutAndReduceShapes(m.shape,f),b=r.backend_util.upcastType(m.dtype,"int32"),k=A(n,g,b),N=r.util.sizeFromShape(y),v=n.data.get(k.dataId).values,x=n.data.get(m.dataId).values;for(let w=0;w<v.length;++w){let T=w*N,S=0;for(let I=0;I<N;++I)S+=x[T+I];v[w]=S}if(u){let _=r.backend_util.expandShapeToKeepDim(k.shape,h),E=k;k=L({inputs:{x:k},backend:n,attrs:{shape:_}}),n.disposeIntermediateTensorInfo(E)}return n.disposeIntermediateTensorInfo(l),null!=d&&n.disposeIntermediateTensorInfo(m),k}let ti={kernelName:r.Sum,backendName:"cpu",kernelFunc:ts},to={kernelName:r.Einsum,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{equation:s}=a,{allDims:i,summedDims:o,idDims:u}=r.backend_util.decodeEinsumEquation(s,t.length);r.backend_util.checkEinsumDimSizes(i.length,u,t);let{path:l,steps:p}=r.backend_util.getEinsumComputePath(o,u),c=p.length,h=null,d=i.length,f=[];for(let m=0;m<c;++m){for(let g of p[m]){let{permutationIndices:y,expandDims:b}=r.backend_util.getEinsumPermutation(d,u[g]),k;r.backend_util.isIdentityPermutation(y)?k=t[g]:(k=J({inputs:{x:t[g]},backend:n,attrs:{perm:y}}),f.push(k));let N=k.shape.slice();for(let v=0;v<b.length;++v)N.splice(b[v],0,1);r.util.arraysEqual(k.shape,N)||(k=L({inputs:{x:k},backend:n,attrs:{shape:N}}),f.push(k)),null===h?h=k:(h=tr({inputs:{a:k,b:h},backend:n}),f.push(h))}m<c-1&&(l[m]>=0&&(h=ts({inputs:{x:h},backend:n,attrs:{axis:l[m]-(i.length-d),keepDims:!1}}),f.push(h)),d--)}for(let x of f)x!==h&&n.disposeIntermediateTensorInfo(x);return h}},tu={kernelName:r.EluGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{dy:s,y:i}=t;a([s,i],"eluGrad");let o=new Float32Array(r.util.sizeFromShape(i.shape)),u=n.data.get(i.dataId).values,l=n.data.get(s.dataId).values;for(let p=0;p<u.length;++p){let c=u[p];c>=1?o[p]=l[p]:o[p]=l[p]*(c+1)}return n.makeTensorInfo(i.shape,"float32",o)}},tl=m((e,t)=>e===t?1:0),tp=B(r.Equal,tl,null,"bool"),tc={kernelName:r.Equal,backendName:"cpu",kernelFunc:tp},th=r.backend_util.ERF_P,td=r.backend_util.ERF_A1,tf=r.backend_util.ERF_A2,tm=r.backend_util.ERF_A3,tg=r.backend_util.ERF_A4,ty=r.backend_util.ERF_A5,tb=o(r.Erf,e=>{let t=Math.sign(e),n=Math.abs(e),r=1/(1+th*n);return t*(1-((((ty*r+tg)*r+tm)*r+tf)*r+td)*r*Math.exp(-n*n))}),tk={kernelName:r.Erf,backendName:"cpu",kernelFunc:tb},tN=w(e=>Math.exp(e)),tv=u(r.Exp,tN,"float32"),tx={kernelName:r.Exp,backendName:"cpu",kernelFunc:tv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function tw(e){let{inputs:t,backend:n,attrs:a}=e,{input:s}=t,{dim:i}=a,o=s.shape.length,u=s.shape.slice(),l=i;return i<0&&(r.util.assert(-(o+1)<=i,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),l=o+i+1),u.splice(l,0,1),L({inputs:{x:s},backend:n,attrs:{shape:u}})}let tT={kernelName:r.ExpandDims,backendName:"cpu",kernelFunc:tw},tS=w(e=>Math.expm1(e)),tI=u(r.Expm1,tS),t_={kernelName:r.Expm1,backendName:"cpu",kernelFunc:tI},tE=m((e,t)=>e/t),tA=B(r.RealDiv,tE),tM={kernelName:r.RealDiv,backendName:"cpu",kernelFunc:tA},tD=m((e,t)=>e-t),t$=O((e,t,n,r)=>({real:e-n,imag:t-r})),tF=B(r.Sub,tD,t$),tB={kernelName:r.Sub,backendName:"cpu",kernelFunc:tF};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function tO(e,t,n){let a=e.shape,s=a[0],i=a[1],o=n.data.get(e.dataId),u=o.complexTensorInfos.real,l=o.complexTensorInfos.imag,p=[s,i],c=r.util.sizeFromShape(p),h=r.util.getTypedArrayFromDType("float32",c),d=r.util.getTypedArrayFromDType("float32",c);for(let f=0;f<s;f++){let m=eT({inputs:{x:u},backend:n,attrs:{begin:[f,0],size:[1,i]}}),g=eT({inputs:{x:l},backend:n,attrs:{begin:[f,0],size:[1,i]}}),y=_({inputs:{real:m,imag:g},backend:n}),{real:b,imag:k}=tR(y,t,n),N=r.backend_util.mergeRealAndImagArrays(b,k);for(let v=0;v<i;v++){let x=r.backend_util.getComplexWithIndex(N,v);h[f*i+v]=x.real,d[f*i+v]=x.imag}n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(y)}let w=n.makeTensorInfo(p,"float32",h),T=n.makeTensorInfo(p,"float32",d),S=_({inputs:{real:w,imag:T},backend:n});return n.disposeIntermediateTensorInfo(w),n.disposeIntermediateTensorInfo(T),S}function tR(e,t,n){var a;let s=r.util.sizeFromShape(e.shape),i=n.data.get(e.dataId),o=n.data.get(i.complexTensorInfos.real.dataId).values,u=n.data.get(i.complexTensorInfos.imag.dataId).values;if(a=s,(a&a-1)==0){let l=function e(t,n,a,s,i){if(1===a)return{real:t,imag:n};let o=r.backend_util.mergeRealAndImagArrays(t,n),u=a/2,l=r.backend_util.complexWithEvenIndex(o),p=l.real,c=l.imag,h=[p.length],d=i.makeTensorInfo(h,"float32",p),f=i.makeTensorInfo(h,"float32",c),m=_({inputs:{real:d,imag:f},backend:i}),g=r.backend_util.complexWithOddIndex(o),y=g.real,b=g.imag,k=[y.length],N=i.makeTensorInfo(k,"float32",y),v=i.makeTensorInfo(k,"float32",b),x=_({inputs:{real:N,imag:v},backend:i}),w=e(p,c,u,s,i),T=w.real,S=w.imag,I=[T.length],E=i.makeTensorInfo(I,"float32",T),A=i.makeTensorInfo(I,"float32",S),D=_({inputs:{real:E,imag:A},backend:i}),$=e(y,b,u,s,i),F=$.real,B=$.imag,O=[F.length],R=i.makeTensorInfo(O,"float32",F),C=i.makeTensorInfo(O,"float32",B),P=_({inputs:{real:R,imag:C},backend:i}),L=r.backend_util.exponents(a,s),z=[L.real.length],W=i.makeTensorInfo(z,"float32",L.real),U=i.makeTensorInfo(z,"float32",L.imag),G=_({inputs:{real:W,imag:U},backend:i}),q=tr({inputs:{a:G,b:P},backend:i}),H=V({inputs:{a:D,b:q},backend:i}),j=tF({inputs:{a:D,b:q},backend:i}),K=M({inputs:{input:H},backend:i}),X=M({inputs:{input:j},backend:i}),Z=eC({inputs:{input:H},backend:i}),Q=eC({inputs:{input:j},backend:i}),Y=eP({inputs:[K,X],backend:i,attrs:{axis:0}}),J=eP({inputs:[Z,Q],backend:i,attrs:{axis:0}}),ee=i.data.get(Y.dataId).values,et=i.data.get(J.dataId).values;return i.disposeIntermediateTensorInfo(d),i.disposeIntermediateTensorInfo(f),i.disposeIntermediateTensorInfo(m),i.disposeIntermediateTensorInfo(N),i.disposeIntermediateTensorInfo(v),i.disposeIntermediateTensorInfo(x),i.disposeIntermediateTensorInfo(E),i.disposeIntermediateTensorInfo(A),i.disposeIntermediateTensorInfo(D),i.disposeIntermediateTensorInfo(R),i.disposeIntermediateTensorInfo(C),i.disposeIntermediateTensorInfo(P),i.disposeIntermediateTensorInfo(W),i.disposeIntermediateTensorInfo(U),i.disposeIntermediateTensorInfo(G),i.disposeIntermediateTensorInfo(q),i.disposeIntermediateTensorInfo(H),i.disposeIntermediateTensorInfo(j),i.disposeIntermediateTensorInfo(K),i.disposeIntermediateTensorInfo(Z),i.disposeIntermediateTensorInfo(X),i.disposeIntermediateTensorInfo(Q),i.disposeIntermediateTensorInfo(Y),i.disposeIntermediateTensorInfo(J),{real:ee,imag:et}}(o,u,s,t,n),p=[e.shape[0],e.shape[1]];if(t){let h=n.makeTensorInfo(p,"float32",l.real),d=n.makeTensorInfo(p,"float32",l.imag),f=n.makeTensorInfo([],"float32",r.util.createScalarValue(s,"float32")),m=c({inputs:{x:f},backend:n}),g=tM.kernelFunc({inputs:{a:h,b:f},backend:n}),y=tM.kernelFunc({inputs:{a:d,b:m},backend:n}),b=n.data.get(g.dataId).values,k=n.data.get(y.dataId).values;return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(y),{real:b,imag:k}}return l}{let N=r.backend_util.mergeRealAndImagArrays(o,u),v=function(e,t,n){let a=new Float32Array(2*t);for(let s=0;s<t;s++){let i=0,o=0;for(let u=0;u<t;u++){let l=r.backend_util.exponent(s*u,t,n),p=r.backend_util.getComplexWithIndex(e,u);i+=p.real*l.real-p.imag*l.imag,o+=p.real*l.imag+p.imag*l.real}n&&(i/=t,o/=t),r.backend_util.assignToTypedArray(a,i,o,s)}return a}(N,s,t);return r.backend_util.splitRealAndImagArrays(v)}}let tC={kernelName:r.FFT,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{input:a}=t,s=r.util.sizeFromShape(a.shape),i=a.shape[a.shape.length-1],o=L({inputs:{x:a},backend:n,attrs:{shape:[s/i,i]}}),u=tO(o,!1,n),l=L({inputs:{x:u},backend:n,attrs:{shape:a.shape}});return n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(u),l}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function tV(e){let{backend:t,attrs:n}=e,{shape:a,value:s,dtype:i}=n,o=i||r.util.inferDtype(s),u=r.util.getArrayFromDType(o,r.util.sizeFromShape(a));return function(e,t,n){e.fill(t)}(u,s,o),t.makeTensorInfo(a,o,u)}let tP={kernelName:r.Fill,backendName:"cpu",kernelFunc:tV},tL={kernelName:r.FlipLeftRight,backendName:"cpu",kernelFunc({inputs:e,attrs:t,backend:n}){let{image:a}=e,s=r.util.getTypedArrayFromDType(a.dtype,r.util.sizeFromShape(a.shape)),[i,o,u,l]=a.shape,p=n.data.get(a.dataId).values;for(let c=0;c<i;c++){let h=c*u*o*l;for(let d=0;d<o;d++){let f=d*(u*l);for(let m=0;m<u;m++){let g=m*l;for(let y=0;y<l;y++){let b=Math.round(u-m-1),k=h+f+g+y,N=p[k];if(b>=0&&b<u){let v=b*l,x=h+f+v+y;N=p[x]}s[k]=N}}}}let w=n.write(s,a.shape,a.dtype);return{dataId:w,shape:a.shape,dtype:a.dtype}}},tz=w(e=>Math.floor(e)),tW=u(r.Floor,tz),tU={kernelName:r.Floor,backendName:"cpu",kernelFunc:tW},tG=m((e,t)=>Math.floor(e/t)),tq=B(r.FloorDiv,tG,null,"int32"),tH={kernelName:r.FloorDiv,backendName:"cpu",kernelFunc:tq},tj={kernelName:r.FusedConv2D,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:r}=e,{x:a,filter:s,bias:i,preluActivationWeights:o}=t,{strides:u,pad:l,dataFormat:p,dilations:c,dimRoundingMode:h,activation:d,leakyreluAlpha:f}=r,m=ez({inputs:{x:a,filter:s},backend:n,attrs:{strides:u,pad:l,dataFormat:p,dilations:c,dimRoundingMode:h}});if(i){let g=m;if("NCHW"===p&&1===i.shape.length&&1!==i.shape[0]){let y=L({inputs:{x:i},backend:n,attrs:{shape:[i.shape[0],1,1]}});m=V({inputs:{a:m,b:y},backend:n}),n.disposeIntermediateTensorInfo(y)}else m=V({inputs:{a:m,b:i},backend:n});n.disposeIntermediateTensorInfo(g)}if(d){let b=m;if("NCHW"===p&&"prelu"===d&&1===o.shape.length&&1!==o.shape[0]){let k=L({inputs:{x:o},backend:n,attrs:{shape:[o.shape[0],1,1]}});m=I(n,m,d,k,f),n.disposeIntermediateTensorInfo(k)}else m=I(n,m,d,o,f);n.disposeIntermediateTensorInfo(b)}return m}},tK={kernelName:r.FusedDepthwiseConv2D,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:r}=e,{x:a,filter:s,bias:i,preluActivationWeights:o}=t,{strides:u,pad:l,dataFormat:p,dilations:c,dimRoundingMode:h,activation:d,leakyreluAlpha:f}=r,m=e3({inputs:{x:a,filter:s},backend:n,attrs:{strides:u,pad:l,dataFormat:p,dilations:c,dimRoundingMode:h}});if(i){let g=m;m=V({inputs:{a:m,b:i},backend:n}),n.disposeIntermediateTensorInfo(g)}if(d){let y=m;m=I(n,m,d,o,f),n.disposeIntermediateTensorInfo(y)}return m}},tX={kernelName:r.GatherNd,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{params:a,indices:s}=t,i=r.util.sizeFromShape(a.shape),o=s.shape,u=o[o.length-1],[l,p,c,h]=r.backend_util.prepareAndValidate(a,s);if(0===p)return n.makeTensorInfo(l,a.dtype,[]);let d=n.data.get(s.dataId).values,f=n.bufferSync(a),m=/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a,s,i,o,u,l){let p=(0,r.buffer)([a,i],n);for(let c=0;c<a;c++){let h=[],d=0;for(let f=0;f<s;f++){let m=e[c*s+f];d+=m*o[f],h.push(m)}if(d<0||d>=l/i)throw Error(`Invalid indices: ${h} does not index into ${u}`);for(let g=0;g<i;g++)p.values[c*i+g]=t.get(...t.indexToLoc(d*i+g))}return p}(d,f,a.dtype,p,u,c,h,a.shape,i);return n.makeTensorInfo(l,a.dtype,m.values)}},tZ={kernelName:r.GatherV2,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,indices:o}=t,{axis:u,batchDims:l}=s;a([i,o],"gatherV2");let p=r.util.parseAxisParam(u,i.shape)[0],c=n.data.get(o.dataId).values,h=i.shape[p];for(let d=0;d<c.length;++d){let f=c[d];r.util.assert(f<=h-1&&f>=0,()=>`GatherV2: the index value ${f} is not in [0, ${h-1}]`)}let m=l;null==l&&(m=0);let g=r.util.sizeFromShape(o.shape),y=r.backend_util.segment_util.collectGatherOpShapeInfo(i,o,p,m),b=L({inputs:{x:i},backend:n,attrs:{shape:[y.batchSize,y.outerSize,y.dimSize,y.sliceSize]}}),k=L({inputs:{x:o},backend:n,attrs:{shape:[y.batchSize,g/y.batchSize]}}),N=[y.batchSize,y.outerSize,g/y.batchSize,y.sliceSize],v=n.bufferSync(k),x=n.bufferSync(b),w=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let a=(0,r.buffer)(n,e.dtype);for(let s=0;s<a.size;++s){let i=a.indexToLoc(s),o=i.slice(),u=o[0],l=o[2],p=t.locToIndex([u,l]);o[2]=t.values[p];let c=e.locToIndex(o);0<=c&&c<e.values.length&&(a.values[s]=e.values[c])}return a}(x,v,N);return n.disposeIntermediateTensorInfo(b),n.disposeIntermediateTensorInfo(k),n.makeTensorInfo(y.outputShape,w.dtype,w.values)}},tQ=m((e,t)=>e>t?1:0),tY=B(r.Greater,tQ,null,"bool"),tJ={kernelName:r.Greater,backendName:"cpu",kernelFunc:tY},t0=m((e,t)=>e>=t?1:0),t1=B(r.GreaterEqual,t0,null,"bool"),t2={kernelName:r.GreaterEqual,backendName:"cpu",kernelFunc:t1},t3={kernelName:r.IFFT,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{input:a}=t,s=r.util.sizeFromShape(a.shape),i=a.shape[a.shape.length-1],o=L({inputs:{x:a},backend:n,attrs:{shape:[s/i,i]}}),u=tO(o,!0,n),l=L({inputs:{x:u},backend:n,attrs:{shape:a.shape}});return n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(u),l}},t6=o(r.IsFinite,e=>Number.isFinite(e)?1:0,"bool"),t4={kernelName:r.IsFinite,backendName:"cpu",kernelFunc:t6},t5=o(r.IsInf,e=>Math.abs(e)===1/0?1:0,"bool"),t8={kernelName:r.IsInf,backendName:"cpu",kernelFunc:t5},t7=o(r.IsNan,e=>Number.isNaN(e)?1:0,"bool"),t9={kernelName:r.IsNan,backendName:"cpu",kernelFunc:t7},ne=m((e,t)=>e<t?1:0),nt=B(r.Less,ne,null,"bool"),nn={kernelName:r.Less,backendName:"cpu",kernelFunc:nt},nr=m((e,t)=>e<=t?1:0),na=B(r.LessEqual,nr,null,"bool"),ns={kernelName:r.LessEqual,backendName:"cpu",kernelFunc:na},ni={kernelName:r.LinSpace,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{backend:t,attrs:n}=e,{start:a,stop:s,num:i}=n,o=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let a=(t-e)/(n-1),s=r.util.makeZerosTypedArray(n,"float32");s[0]=e;for(let i=1;i<s.length;i++)s[i]=s[i-1]+a;return s}(a,s,i);return t.makeTensorInfo([o.length],"float32",o)}},no=w(e=>Math.log(e)),nu=u(r.Log,no),nl={kernelName:r.Log,backendName:"cpu",kernelFunc:nu},np=o(r.Log1p,e=>Math.log1p(e)),nc={kernelName:r.Log1p,backendName:"cpu",kernelFunc:np},nh=m((e,t)=>e&&t),nd=B(r.LogicalAnd,nh,null,"bool"),nf={kernelName:r.LogicalAnd,backendName:"cpu",kernelFunc:nd},nm=o(r.LogicalNot,e=>e?0:1,"bool"),ng={kernelName:r.LogicalNot,backendName:"cpu",kernelFunc:nm},ny=m((e,t)=>e||t),nb=B(r.LogicalOr,ny,null,"bool"),nk={kernelName:r.LogicalOr,backendName:"cpu",kernelFunc:nb},nN={kernelName:r.LRN,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{depthRadius:o,bias:u,alpha:l,beta:p}=s;a(i,"LRN");let c=i.shape[3],h=c-1,d=n.data.get(i.dataId).values,f=r.util.sizeFromShape(i.shape),m=new Float32Array(f);function g(e){let t=e%c,n=e-t+Math.max(0,t-o),r=e-t+Math.min(t+o,h),a=0;for(;n<=r;n++){let s=d[n];a+=s*s}return a}for(let y=0;y<f;y++){let b=g(y),k=d[y]*Math.pow(u+l*b,-p);m[y]=k}return n.makeTensorInfo(i.shape,i.dtype,m)}},nv={kernelName:r.LRNGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,y:o,dy:u}=t,{depthRadius:l,bias:p,alpha:c,beta:h}=s;a(u,"LRNGrad");let d=r.util.sizeFromShape(u.shape),f=u.shape[3],m=n.data.get(u.dataId).values,g=n.data.get(i.dataId).values,y=n.data.get(o.dataId).values,b=new Float32Array(d);for(let k=0;k<d;k++){let N=k%f,v=k-N+Math.max(0,N-l),x=k-N+Math.min(f,N+l+1),w=0;for(let T=v;T<x;T++)w+=Math.pow(g[T],2);w=c*w+p;for(let S=v;S<x;S++){let I=-2*c*h*g[S]*y[k]/w;k===S&&(I+=Math.pow(w,-h)),I*=m[k],b[S]+=I}}return n.makeTensorInfo(u.shape,i.dtype,b)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nx(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{reductionIndices:o,keepDims:u}=s,l=i.shape,p=l.length,c=r.util.parseAxisParam(o,l),h=c,d=r.backend_util.getAxesPermutation(h,p),f=n.data.get(i.dataId).values;if(null!=d){let m=Array(p);for(let g=0;g<m.length;g++)m[g]=l[d[g]];f=Y(f,l,i.dtype,d,m),h=r.backend_util.getInnerMostAxes(h.length,p),l=m}a(i,"max"),r.backend_util.assertAxesAreInnerMostDims("max",h,p);let[y,b]=r.backend_util.computeOutAndReduceShapes(l,h),k=r.util.sizeFromShape(b),N=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a){let s=r.util.getTypedArrayFromDType(a,r.util.sizeFromShape(n));for(let i=0;i<s.length;++i){let o=i*t,u=e[o];for(let l=0;l<t;++l){let p=e[o+l];(Number.isNaN(p)||p>u)&&(u=p)}s[i]=u}return s}(f,k,y,i.dtype),v=n.write(N,y,i.dtype),x=y;if(u){let w=r.backend_util.expandShapeToKeepDim(y,c);x=w}return{dataId:v,shape:x,dtype:i.dtype}}let nw={kernelName:r.Max,backendName:"cpu",kernelFunc:nx},nT=m((e,t)=>Math.max(e,t)),nS=B(r.Maximum,nT),nI={kernelName:r.Maximum,backendName:"cpu",kernelFunc:nS},n_={kernelName:r.MaxPool,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t;a(i,"maxPool");let{filterSize:o,strides:u,pad:l,dimRoundingMode:p}=s;r.util.assert(r.backend_util.eitherStridesOrDilationsAreOne(u,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${u} and dilations '1'`);let h=r.backend_util.computePool2DInfo(i.shape,o,u,1,l,p),d;if(1===h.filterWidth&&1===h.filterHeight&&r.util.arraysEqual(h.inShape,h.outShape))d=c({inputs:{x:i},backend:n});else{let f=n.data.get(i.dataId).values,m=r.util.computeStrides(i.shape),g=eg(f,i.shape,i.dtype,m,h,"max");d=n.makeTensorInfo(h.outShape,i.dtype,g.values)}return d}},nE={kernelName:r.MaxPool3D,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{filterSize:o,strides:u,pad:l,dimRoundingMode:p,dataFormat:c}=s;a(i,"maxPool3d");let h=r.backend_util.computePool3DInfo(i.shape,o,u,1,l,p,c),d=n.data.get(i.dataId).values,f=eb(d,i.shape,i.dtype,r.util.computeStrides(i.shape),h,"max");return n.makeTensorInfo(f.shape,"float32",f.values)}},nA={kernelName:r.MaxPool3DGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{dy:i,input:o}=t,{filterSize:u,strides:l,pad:p,dimRoundingMode:c}=s;a([i,o],"maxPool3DGrad");let h=r.backend_util.computePool3DInfo(o.shape,u,l,1,p,c),d=n.bufferSync(o),f=function(e,t){let n=(0,r.buffer)(t.outShape,"int32"),a=t.strideDepth,s=t.strideHeight,i=t.strideWidth,o=t.dilationDepth,u=t.dilationHeight,l=t.dilationWidth,p=t.effectiveFilterDepth,c=t.effectiveFilterHeight,h=t.effectiveFilterWidth,d=t.padInfo.front,f=t.padInfo.top,m=t.padInfo.left;for(let g=0;g<t.batchSize;++g)for(let y=0;y<t.inChannels;++y)for(let b=0;b<t.outDepth;++b){let k=b*a-d,N=k;for(;N<0;)N+=o;let v=Math.min(t.inDepth,p+k);for(let x=0;x<t.outHeight;++x){let w=x*s-f,T=w;for(;T<0;)T+=u;let S=Math.min(t.inHeight,c+w);for(let I=0;I<t.outWidth;++I){let _=I*i-m,E=_;for(;E<0;)E+=l;let A=Math.min(t.inWidth,h+_),M=Number.NEGATIVE_INFINITY,D=-1;for(let $=N;$<v;$+=o){let F=$-k;for(let B=T;B<S;B+=u){let O=B-w;for(let R=E;R<A;R+=l){let C=R-_,V=e.get(g,$,B,R,y);V>=M&&(M=V,D=F*c*h+O*c+C)}}}n.set(D,g,b,x,I,y)}}}return n}(d,h),m=h.strideDepth,g=h.strideHeight,y=h.strideWidth,b=h.dilationDepth,k=h.dilationHeight,N=h.dilationWidth,v=h.effectiveFilterDepth,x=h.effectiveFilterHeight,w=h.effectiveFilterWidth,T=v-1-h.padInfo.front,S=w-1-h.padInfo.left,I=x-1-h.padInfo.top,_=(0,r.buffer)(o.shape,"float32"),E=n.bufferSync(i);for(let A=0;A<h.batchSize;++A)for(let M=0;M<h.inChannels;++M)for(let D=0;D<h.inDepth;++D)for(let $=0;$<h.inHeight;++$)for(let F=0;F<h.inWidth;++F){let B=D-T,O=$-I,R=F-S,C=0;for(let V=0;V<v;V+=b){let P=(B+V)/m;if(!(P<0)&&!(P>=h.outDepth)&&Math.floor(P)===P)for(let L=0;L<x;L+=k){let z=(O+L)/g;if(!(z<0)&&!(z>=h.outHeight)&&Math.floor(z)===z)for(let W=0;W<w;W+=N){let U=(R+W)/y;if(U<0||U>=h.outWidth||Math.floor(U)!==U)continue;let G=v*x*w-1-f.get(A,P,z,U,M),q=V*x*w+L*w+W,H=G===q?1:0;if(0===H)continue;let j=E.get(A,P,z,U,M);C+=j*H}}}_.set(C,A,D,$,F,M)}return n.makeTensorInfo(_.shape,_.dtype,_.values)}},nM={kernelName:r.MaxPoolGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{dy:i,input:o,output:u}=t;a([o,u],"maxPoolGrad");let{filterSize:l,strides:p,pad:c,dimRoundingMode:h}=s,d=r.backend_util.computePool2DInfo(o.shape,l,p,1,c,h),f=n.data.get(o.dataId).values,m=(0,r.buffer)(d.outShape,o.dtype,ey(f,o.shape,o.dtype,d).values),g=d.strideHeight,y=d.strideWidth,b=d.dilationHeight,k=d.dilationWidth,N=d.effectiveFilterHeight,v=d.effectiveFilterWidth,x=v-1-d.padInfo.left,w=N-1-d.padInfo.top,T=(0,r.buffer)(o.shape,"float32"),S=n.data.get(i.dataId).values,I=(0,r.buffer)(i.shape,"float32",S);for(let _=0;_<d.batchSize;++_)for(let E=0;E<d.inChannels;++E)for(let A=0;A<d.inHeight;++A)for(let M=0;M<d.inWidth;++M){let D=A-w,$=M-x,F=0;for(let B=0;B<N;B+=b){let O=(D+B)/g;if(!(O<0)&&!(O>=d.outHeight)&&Math.floor(O)===O)for(let R=0;R<v;R+=k){let C=($+R)/y;if(C<0||C>=d.outWidth||Math.floor(C)!==C)continue;let V=N*v-1-m.get(_,O,C,E),P=B*v+R,L=V===P?1:0;if(0===L)continue;let z=I.get(_,O,C,E);F+=z*L}}T.set(F,_,A,M,E)}return n.makeTensorInfo(T.shape,T.dtype,T.values)}},nD={kernelName:r.MaxPoolWithArgmax,backendName:"cpu",kernelFunc({inputs:e,attrs:t,backend:n}){let{x:s}=e,{filterSize:i,strides:o,pad:u,includeBatchInIndex:l}=t;a(s,"MaxPoolWithArgmax");let p=n.data.get(s.dataId).values,c=r.backend_util.computePool2DInfo(s.shape,i,o,[1,1],u),[h,d]=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a,s){let i=r.util.computeStrides(t),o=eg(e,t,n,i,s,"max"),u=ey(e,t,n,s,!0,a);return[o.values,u.values]}(p,s.shape,s.dtype,l,c),f=n.write(h,c.outShape,s.dtype),m=n.write(d,c.outShape,s.dtype);return[{dataId:f,shape:c.outShape,dtype:s.dtype},{dataId:m,shape:c.outShape,dtype:"int32"}]}},n$={kernelName:r.Mean,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{x:s}=t,{axis:i,keepDims:o}=a,u=r.util.parseAxisParam(i,s.shape),l=r.backend_util.computeOutAndReduceShapes(s.shape,u),p=l[1],c=r.util.sizeFromShape(p),h=[],d=n.makeTensorInfo([],"float32",new Float32Array([c]));h.push(d);let f=$({inputs:{x:s},backend:n,attrs:{dtype:"float32"}});h.push(f);let m=tA({inputs:{a:f,b:d},backend:n});h.push(m);let g=ts({inputs:{x:m},backend:n,attrs:{axis:i,keepDims:o}});return h.forEach(e=>n.disposeIntermediateTensorInfo(e)),g}},nF={kernelName:r.Min,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o,keepDims:u}=s;a(i,"min");let l=r.util.parseAxisParam(o,i.shape),p=l,c=r.backend_util.getAxesPermutation(p,i.shape.length),h=i;null!=c&&(h=J({inputs:{x:i},backend:n,attrs:{perm:c}}),p=r.backend_util.getInnerMostAxes(p.length,i.shape.length)),r.backend_util.assertAxesAreInnerMostDims("min",p,h.shape.length);let[d,f]=r.backend_util.computeOutAndReduceShapes(h.shape,p),m=r.util.sizeFromShape(f),g=r.util.makeZerosTypedArray(r.util.sizeFromShape(d),h.dtype),y=n.data.get(h.dataId).values;for(let b=0;b<g.length;++b){let k=b*m,N=y[k];for(let v=0;v<m;++v){let x=y[k+v];(Number.isNaN(x)||x<N)&&(N=x)}g[b]=N}null!=c&&n.disposeIntermediateTensorInfo(h);let w=n.makeTensorInfo(d,h.dtype,g);if(u){let T=r.backend_util.expandShapeToKeepDim(d,l),S=L({inputs:{x:w},backend:n,attrs:{shape:T}});return n.disposeIntermediateTensorInfo(w),S}return w}},nB=m((e,t)=>Math.min(e,t)),nO=B(r.Minimum,nB),nR={kernelName:r.Minimum,backendName:"cpu",kernelFunc:nO},nC={kernelName:r.MirrorPad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{paddings:o,mode:u}=s;a(i,"mirrorPad");let l=o.map((e,t)=>e[0]+i.shape[t]+e[1]),p=o.map(e=>e[0]),c=o.map((e,t)=>e[0]+i.shape[t]),h="reflect"===u?0:1,d=n.data.get(i.dataId).values,f=i.shape.length,m=r.util.computeStrides(i.shape),g=r.util.sizeFromShape(l),y=l.length,b=r.util.computeStrides(l),k=r.util.getTypedArrayFromDType(i.dtype,g);for(let N=0;N<g;N++){let v=r.util.indexToLoc(N,y,b);for(let x=0;x<y;x++)v[x]<p[x]?v[x]=2*p[x]-v[x]-h:v[x]>=c[x]&&(v[x]=(c[x]-1)*2-v[x]+h);v=v.map((e,t)=>e-p[t]);let w=r.util.locToIndex(v,f,m);k[N]=d[w]}let T=n.write(k,l,i.dtype);return{dataId:T,shape:l,dtype:i.dtype}}},nV=m((e,t)=>{let n=e%t;return e<0&&t<0||e>=0&&t>=0?n:(n+t)%t}),nP=B(r.Mod,nV),nL={kernelName:r.Mod,backendName:"cpu",kernelFunc:nP};var nz=n(6377);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nW(e){let{inputs:t,backend:n,attrs:a}=e,{logits:s}=t,{dim:i}=a,o=s.shape.length,u=i;if(-1===u&&(u=o-1),u!==o-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${o} and dim was ${u}`);let l=r.util.parseAxisParam([u],s.shape),p=nx({inputs:{x:s},backend:n,attrs:{reductionIndices:l,keepDims:!1}}),c=r.backend_util.expandShapeToKeepDim(p.shape,l),h=L({inputs:{x:p},backend:n,attrs:{shape:c}}),d=tF({inputs:{a:s,b:h},backend:n}),f=tv({inputs:{x:d},backend:n}),m=ts({inputs:{x:f},backend:n,attrs:{axis:l,keepDims:!1}}),g=L({inputs:{x:m},backend:n,attrs:{shape:c}}),y=tA({inputs:{a:f,b:g},backend:n});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(g),y}let nU={kernelName:r.Softmax,backendName:"cpu",kernelFunc:nW},nG={kernelName:r.Multinomial,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{logits:i}=t,{numSamples:o,seed:u,normalized:l}=s;a(i,"multinomial");let p=l?i:nW({inputs:{logits:i},backend:n,attrs:{dim:-1}}),c=p.shape[0],h=p.shape[1],d=n.data.get(p.dataId).values,f=[c,o],m=r.util.makeZerosTypedArray(r.util.sizeFromShape(f),"int32");for(let g=0;g<c;++g){let y=g*h,b=new Float32Array(h-1);b[0]=d[y];for(let k=1;k<b.length;++k)b[k]=b[k-1]+d[y+k];let N=nz.alea(u.toString()),v=g*o;for(let x=0;x<o;++x){let w=N();m[v+x]=b.length;for(let T=0;T<b.length;T++)if(w<b[T]){m[v+x]=T;break}}}return l||n.disposeIntermediateTensorInfo(p),n.makeTensorInfo(f,"int32",m)}},nq={kernelName:r.Neg,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:n}=e,{x:s}=t;a(s,"neg");let i=n.data.get(s.dataId).values,[o,u]=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let a=r.util.createScalarValue(-1,n);return tt([],t,a,e,n)}(i,s.shape,s.dtype);return n.makeTensorInfo(u,s.dtype,o)}},nH=r.kernel_impls.nonMaxSuppressionV3Impl,nj={kernelName:r.NonMaxSuppressionV3,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:s,scores:i}=t,{maxOutputSize:o,iouThreshold:u,scoreThreshold:l}=r;a(s,"NonMaxSuppression");let p=n.data.get(s.dataId).values,c=n.data.get(i.dataId).values,{selectedIndices:h}=nH(p,c,o,u,l);return n.makeTensorInfo([h.length],"int32",new Int32Array(h))}},nK=r.kernel_impls.nonMaxSuppressionV4Impl,nX={kernelName:r.NonMaxSuppressionV4,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:s,scores:i}=t,{maxOutputSize:o,iouThreshold:u,scoreThreshold:l,padToMaxOutputSize:p}=r;a(s,"NonMaxSuppressionPadded");let c=n.data.get(s.dataId).values,h=n.data.get(i.dataId).values,{selectedIndices:d,validOutputs:f}=nK(c,h,o,u,l,p);return[n.makeTensorInfo([d.length],"int32",new Int32Array(d)),n.makeTensorInfo([],"int32",new Int32Array([f]))]}},nZ=r.kernel_impls.nonMaxSuppressionV5Impl,nQ={kernelName:r.NonMaxSuppressionV5,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:s,scores:i}=t,{maxOutputSize:o,iouThreshold:u,scoreThreshold:l,softNmsSigma:p}=r;a(s,"NonMaxSuppressionWithScore");let c=n.data.get(s.dataId).values,h=n.data.get(i.dataId).values,{selectedIndices:d,selectedScores:f}=nZ(c,h,o,u,l,p);return[n.makeTensorInfo([d.length],"int32",new Int32Array(d)),n.makeTensorInfo([f.length],"float32",new Float32Array(f))]}},nY=m((e,t)=>e!==t?1:0),nJ=B(r.NotEqual,nY,null,"bool"),n0={kernelName:r.NotEqual,backendName:"cpu",kernelFunc:nJ},n1={kernelName:r.OneHot,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{indices:i}=t,{dtype:o,depth:u,onValue:l,offValue:p}=s;a(i,"oneHot");let c=r.util.sizeFromShape(i.shape),h=new Float32Array(c*u);h.fill(p);let d=n.data.get(i.dataId).values;for(let f=0;f<c;++f)d[f]>=0&&d[f]<u&&(h[f*u+d[f]]=l);return n.makeTensorInfo([...i.shape,u],o,h)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function n2(e){let{inputs:t,backend:n}=e,{x:r}=t;if("string"===r.dtype)throw Error("zerosLike is not supported for string tensors");if("complex64"!==r.dtype)return tV({backend:n,attrs:{shape:r.shape,value:0,dtype:r.dtype}});{let a=M({inputs:{input:r},backend:n}),s=n2({inputs:{x:a},backend:n}),i=eC({inputs:{input:r},backend:n}),o=n2({inputs:{x:i},backend:n}),u=_({inputs:{real:s,imag:o},backend:n});return n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(o),u}}let n3={kernelName:r.ZerosLike,backendName:"cpu",kernelFunc:n2},n6={kernelName:r.OnesLike,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function e(t){let{inputs:n,backend:r}=t,{x:a}=n;if("string"===a.dtype)throw Error("onesLike is not supported for string tensors");if("complex64"!==a.dtype)return tV({backend:r,attrs:{shape:a.shape,value:1,dtype:a.dtype}});{let s=M({inputs:{input:a},backend:r}),i=e({inputs:{x:s},backend:r}),o=eC({inputs:{input:a},backend:r}),u=n2({inputs:{x:o},backend:r}),l=_({inputs:{real:i,imag:u},backend:r});return r.disposeIntermediateTensorInfo(s),r.disposeIntermediateTensorInfo(i),r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(u),l}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function n4(e){let{inputs:t,backend:n,attrs:a}=e,{axis:s}=a;if(1===t.length)return tw({inputs:{input:t[0]},backend:n,attrs:{dim:s}});let i=t[0].shape,o=t[0].dtype;t.forEach(e=>{r.util.assertShapesMatch(i,e.shape,"All tensors passed to stack must have matching shapes"),r.util.assert(o===e.dtype,()=>"All tensors passed to stack must have matching dtypes")});let u=[],l=t.map(e=>{let t=tw({inputs:{input:e},backend:n,attrs:{dim:s}});return u.push(t),t}),p=eP({inputs:l,backend:n,attrs:{axis:s}});return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}let n5={kernelName:r.Pack,backendName:"cpu",kernelFunc:n4},n8={kernelName:r.PadV2,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{paddings:o,constantValue:u}=s;a(i,"pad");let l=o.map((e,t)=>e[0]+i.shape[t]+e[1]),p=o.map(e=>e[0]),c=n.data.get(i.dataId).values,h=r.util.sizeFromShape(i.shape),d=i.shape.length,f=r.util.computeStrides(i.shape),m=r.util.sizeFromShape(l),g=l.length,y=r.util.computeStrides(l),b=r.util.getTypedArrayFromDType(i.dtype,m);0!==u&&b.fill(u);for(let k=0;k<h;k++){let N=r.util.indexToLoc(k,d,f),v=N.map((e,t)=>e+p[t]),x=r.util.locToIndex(v,g,y);b[x]=c[k]}let w=n.write(b,l,i.dtype);return{dataId:w,shape:l,dtype:i.dtype}}},n7=m((e,t)=>Math.pow(e,t)),n9=B(r.Pow,n7),re={kernelName:r.Pow,backendName:"cpu",kernelFunc:n9},rt={kernelName:r.Prod,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{axis:o,keepDims:u}=s;a(i,"prod");let l=i.shape.length,p=r.util.parseAxisParam(o,i.shape),c=r.backend_util.getAxesPermutation(p,l),h=p,d=i,f=[];null!=c&&(f.push(d=J({inputs:{x:i},backend:n,attrs:{perm:c}})),h=r.backend_util.getInnerMostAxes(h.length,l));let m=n.data.get(d.dataId).values,{outVals:g,outShape:y,outDtype:b}=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a){let[s,i]=r.backend_util.computeOutAndReduceShapes(e,a),o=(0,r.upcastType)(t,"int32"),u=r.util.makeZerosTypedArray(r.util.sizeFromShape(s),o),l=r.util.sizeFromShape(i);for(let p=0;p<u.length;++p){let c=p*l,h=1;for(let d=0;d<l;++d)h*=n[c+d];u[p]=h}return{outVals:u,outShape:s,outDtype:o}}(d.shape,d.dtype,m,h),k=y;return u&&(k=r.backend_util.expandShapeToKeepDim(y,p)),f.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(k,b,g)}};function rn(e,t){let n=e.slice(0,t);for(;n.length<t;)n.push(1);for(let r=t;r<e.length;r++)n[t-1]*=e[r];return n}let rr={kernelName:r.RaggedGather,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{paramsNestedSplits:s,paramsDenseValues:i,indices:o}=t,{outputRaggedRank:u}=a,l=s.map(e=>n.data.get(e.dataId).values),p=s.map(e=>e.shape),c=n.data.get(i.dataId).values,h=n.data.get(o.dataId).values,[d,f,m]=function(e,t,n,a,s,i,o,u){if(0===e.length)throw Error("paramsNestedSplits must be non empty");if(0===t[0].length)throw Error("Split tensors must not be scalars");let l=t[0][0]-1;if(!/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){e.forEach((e,a)=>{if(e<0||e>=n){let s=r.util.indexToLoc(a,t.length,r.util.computeStrides(t)).join(",");throw Error(`indices[${s}] = ${e} is not in [0, ${n})`)}})}(i,o,l),0===a.length)throw Error("params.rank must be nonzero");let p=a[0],{outSplits:c,valueSlices:h,numValues:d}=function(e,t,n,r){let a=[],s=0,i=t.length-1+n.length,o=Array(i).fill(null).map(()=>[0]);!function(e,t){for(let n=0;n<e.length;++n){let r=e[n],a=n===e.length-1?t:e[n+1].length;if(0===r.length)throw Error("Ragged splits may not be empty");if(r[0]<0)throw Error("Ragged splits must be non-negative");if(r[r.length-1]>a)throw Error("Ragged splits must not point past values");for(let s=1;s<r.length;++s)if(r[s-1]>r[s])throw Error("Ragged splits must be sorted in ascending order")}}(n,r);let u=1;for(let l=0;l<t.length-1;++l){u*=t[l];let p=t[l+1];for(let c=1;c<u+1;++c)o[l].push(c*p)}for(let h=0;h<e.length;++h){let d=e[h],f=e[h]+1;for(let m=0;m<n.length;++m){let g=n[m],y=m+t.length-1;if(y>=0){let b=o[y],k=b[b.length-1]-g[d];for(let N=d;N<f;++N)o[y].push(g[N+1]+k)}d=g[d],f=g[f]}f!==d&&(a.push([d,f]),s+=f-d)}return{outSplits:o,valueSlices:a,numValues:s}}(i,o,e,p),f=function(e){let t=[];for(let n=0;n<e.length;++n){let a=e[n].length,s=r.util.getArrayFromDType("int32",a);t.push(s),e[n].forEach((e,t)=>s[t]=e)}return t}(c),m=function(e,t,n,a,s){let i=t.slice();i[0]=s;let o=r.util.getArrayFromDType(n,r.util.sizeFromShape(i)),u=e.length,l=0===u?0:u/t[0];return!function(e,t,n,r,a,s){let i=rn(t,2)[1],o=rn(s,2)[1],u=0;for(let l of n)for(let p=l[0];p<l[1];++p){for(let c=0;c<r;++c)a[u*o+c]=e[p*i+c];++u}}(e,t,a,l,o,i),[o,i]}(n,a,s,h,d);return[f,m[0],m[1]]}(l,p,c,i.shape,i.dtype,h,o.shape,u),g=d.map(e=>n.makeTensorInfo([e.length],"int32",e)),y=n.makeTensorInfo(m,i.dtype,f);return g.concat([y])}},ra={kernelName:r.RaggedRange,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{starts:a,limits:s,deltas:i}=t,o=n.data.get(a.dataId).values,u=n.data.get(s.dataId).values,l=n.data.get(i.dataId).values,[p,c]=function(e,t,n,a,s,i,o){if(t.length>1)throw Error("starts must be a scalar or vector");if(s.length>1)throw Error("limits must be a scalar or vector");if(o.length>1)throw Error("deltas must be a scalar or vector");let u=0===t.length,l=0===s.length,p=0===o.length,c=[];u||c.push(t[0]),l||c.push(s[0]),p||c.push(o[0]);for(let h=1;h<c.length;++h)if(c[h]!==c[h-1])throw Error("starts, limits, and deltas must have the same shape");let d=0===c.length?1:c[0],f=r.util.getArrayFromDType("int32",d+1);f[0]=0;for(let m=0;m<d;++m){let g=u?e[0]:e[m],y=l?a[0]:a[m],b=p?i[0]:i[m];if(0===b)throw Error("Requires delta != 0");let k;if(b>0&&y<g||b<0&&y>g)k=0;else if((k=Math.ceil(Math.abs((y-g)/b)))>2147483647)throw Error("Requires ((limit - start) / delta) <= 2147483647");f[m+1]=f[m]+k}let N=f[d],v=r.util.getArrayFromDType(n,N),x=0;for(let w=0;w<d;++w){let T=f[w+1]-f[w],S=u?e[0]:e[w],I=p?i[0]:i[w];for(let _=0;_<T;++_)v[x++]=S,S+=I}return[f,v]}(o,a.shape,a.dtype,u,s.shape,l,i.shape),h=n.makeTensorInfo([p.length],"int32",p),d=n.makeTensorInfo([c.length],a.dtype,c);return[h,d]}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ var rs=r.backend_util.RowPartitionType;class ri{constructor(e,t,n,a,s,i,o,u,l,p){this.shape=e,this.shapeShape=t,this.values=n,this.valuesShape=a,this.valuesDType=s,this.defaultValue=i,this.defaultValueShape=o,this.rowPartitionValues=u,this.rowPartitionValuesShapes=l,this.rowPartitionTypes=r.backend_util.getRowPartitionTypesHelper(p),this.raggedRank=r.backend_util.getRaggedRank(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===rs.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===rs.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){let t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case rs.VALUE_ROWIDS:return ri.getMaxWidthValueRowID(t);case rs.ROW_SPLITS:return ri.getMaxWidthRowSplit(t);default:throw Error(`Cannot handle partition type ${rs[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){let t=e.length;if(0===t||1===t)return 0;let n=0;for(let r=0;r<t-1;++r){let a=e[r+1]-e[r];a>n&&(n=a)}return n}static getMaxWidthValueRowID(e){let t=e.length;if(0===t)return 0;let n=0,r=e[0],a=0;for(let s=1;s<t;++s){let i=e[s];i!==r&&(r=i,a=Math.max(s-n,a),n=s)}return Math.max(t-n,a)}tensorShapeFromTensor(e,t,n=!0){if(0===t.length){if(-1===e[0])return[];throw Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return ru(e,n)}calculateOutputSize(e){let t=this.valuesShape,n=this.defaultValueShape;r.backend_util.validateDefaultValueShape(n,t);let a=this.tensorShapeFromTensor(this.shape,this.shapeShape),s=r.backend_util.combineRaggedTensorToTensorShapes(this.raggedRank,a,t),i=s;i[0]<0&&(i[0]=e);for(let o=1;o<=this.raggedRank;++o)i[o]<0&&(i[o]=this.getMaxWidth(o));return i}calculateFirstParentOutputIndex(e,t,n){let a=Math.min(e,n),s=[],i=0;for(let o=0;o<a;++o,i+=t)s.push(i);for(let u=a;u<e;++u)s.push(-1);return r.util.assert(s.length===e,()=>"Final length of result must be equal to firstDimension."),s}calculateOutputIndexRowSplit(e,t,n,r){let a=e.length,s=[];for(let i=0;i<a-1;++i){let o=e[i+1]-e[i],u=Math.min(r,o),l=t[i];-1===l&&(u=0);for(let p=0;p<u;++p)s.push(l),l+=n;for(let c=0;c<o-u;++c)s.push(-1)}if(a>0&&s.length!==e[a-1])throw Error("Invalid row split size.");return s}calculateOutputIndexValueRowID(e,t,n,r){let a=e.length,s=[];if(0===a)return[];let i=0,o=e[0];if(o>=t.length)throw Error(`Got currentValueRowId=${o}, which is not less than ${t.length}`);let u=t[o];s.push(u);for(let l=1;l<a;++l){let p=e[l];if(p===o)u>=0&&(++i<r?u+=n:u=-1);else{if(i=0,o=p,p>=t.length)throw Error(`Got nextValueRowId=${p} which is not less than ${t.length}`);u=t[p]}s.push(u)}if(s.length!==e.length)throw Error("Invalid row ids.");return s}calculateOutputIndex(e,t,n,r){let a=this.getRowPartitionTensor(e),s=this.getRowPartitionTypeByDimension(e);switch(s){case rs.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(a,t,n,r);case rs.ROW_SPLITS:if(a.length-1>t.length)throw Error(`Row partition size is greater than output size: ${a.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(a,t,n,r);default:throw Error(`Unsupported partition type: ${rs[s]}`)}}getFirstDimensionSize(){let e=this.rowPartitionValues[0];if(0===this.rowPartitionTypes.length)throw Error("No row_partition_types given.");let t=this.rowPartitionTypes[0];switch(t){case rs.FIRST_DIM_SIZE:return e[0];case rs.VALUE_ROWIDS:throw Error("Cannot handle VALUE_ROWIDS in first dimension.");case rs.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw Error(`Cannot handle type ${rs[t]}`)}}compute(){let e=this.rowPartitionValues[0];if(e.length<=0)throw Error("Invalid first partition input. Tensor requires at least one element.");let t=this.getFirstDimensionSize(),n=this.calculateOutputSize(t),a=Array(this.raggedRank+1);a[a.length-1]=1;for(let s=a.length-2;s>=0;--s)a[s]=a[s+1]*n[s+1];let i=ru(n,!1),o=r.util.getArrayFromDType(this.valuesDType,r.util.sizeFromShape(i)),u=a[0]*n[0];if(u>0){let l=this.calculateFirstParentOutputIndex(t,a[0],n[0]);for(let p=1;p<=this.raggedRank;++p){let c=this.calculateOutputIndex(p-1,l,a[p],n[p]);l=c}this.setOutput(this.raggedRank,l,o,i)}return[i,o]}setOutput(e,t,n,a){if(0===n.length)return;let s=this.values,i=a.slice();i=i.slice(e+1);let o=r.util.sizeFromShape(i),u=t.length,l=this.defaultValue;if(l.length!==o&&1!==l.length){let p=this.defaultValueShape;(0,r.tidy)(()=>{let e=(0,r.reshape)(l,p),t=(0,r.broadcastTo)(e,i);l=t.dataSync()})}let c=0,h=0,d=0;for(let f=0;f<=u;++f){let m=f<u?t[f]:-1;if(m===d){++d;continue}if(h<d){let g=s.subarray(c*o),y=n.subarray(h*o),b=(d-h)*o;ro(y,g,b)}if(f>=u){let k=n.length;m=Math.floor(k/o)}if(m>d){if(1===this.defaultValue.length)n.subarray(d*o,m*o).fill(this.defaultValue[0]),d=m;else for(;m>d;){let N=n.slice(d*o);ro(N,l,o),++d}}m<0?(c=f+1,h=d):(c=f,d=(h=d)+1)}}}function ro(e,t,n){for(let r=0;r<n;r++)e[r]=t[r]}function ru(e,t){let n=[];for(let r of e){if(r<0){if(!t)throw Error(`Dimension ${r} must be >= 0`);if(r<-1)throw Error(`Dimension ${r} must be >= -1`);r=-1}n.push(r)}return n}let rl={kernelName:r.RaggedTensorToTensor,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){var t,n,r,a,s,i,o,u,l,p;let{inputs:c,backend:h,attrs:d}=e,{shape:f,values:m,defaultValue:g,rowPartitionTensors:y}=c,{rowPartitionTypes:b}=d,k=h.data.get(f.dataId).values,N=h.data.get(m.dataId).values,v=h.data.get(g.dataId).values,x=y.map(e=>h.data.get(e.dataId).values),w=y.map(e=>e.shape),[T,S]=(n=f.shape,a=m.shape,s=m.dtype,o=g.shape,new ri(k,n,N,a,s,v,o,x,w,b).compute());return h.makeTensorInfo(T,m.dtype,S)}},rp={kernelName:r.Range,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{backend:t,attrs:n}=e,{start:a,stop:s,dtype:i,step:o}=n,u=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a){let s=e<t&&n<0,i=t<e&&n>1;if(e===t||s||i)return r.util.makeZerosTypedArray(0,a);let o=Math.abs(Math.ceil((t-e)/n)),u=r.util.makeZerosTypedArray(o,a);t<e&&1===n&&(n=-1),u[0]=e;for(let l=1;l<u.length;l++)u[l]=u[l-1]+n;return u}(a,s,o,i);return t.makeTensorInfo([u.length],i,u)}},rc=o(r.Reciprocal,e=>1/e),rh={kernelName:r.Reciprocal,backendName:"cpu",kernelFunc:rc},rd={kernelName:r.ResizeBilinear,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{images:i}=t,{alignCorners:o,halfPixelCenters:u,size:l}=s;a(i,"resizeBilinear");let p=r.util.computeStrides(i.shape),[c,h]=l,[d,f,m,g]=i.shape,y=n.data.get(i.dataId).values,b=new Float32Array(r.util.sizeFromShape([d,c,h,g])),k=[o&&c>1?f-1:f,o&&h>1?m-1:m],N=[o&&c>1?c-1:c,o&&h>1?h-1:h],v=0,x=k[0]/N[0],w=k[1]/N[1];for(let T=0;T<d;T++)for(let S=0;S<c;S++){let I;I=u?x*(S+.5)-.5:x*S;let _=Math.max(0,Math.floor(I)),E=I-_,A=Math.min(f-1,Math.ceil(I)),M=T*p[0]+_*p[1],D=T*p[0]+A*p[1];for(let $=0;$<h;$++){let F;F=u?w*($+.5)-.5:w*$;let B=Math.max(0,Math.floor(F)),O=F-B,R=Math.min(m-1,Math.ceil(F)),C=M+B*p[2],V=D+B*p[2],P=M+R*p[2],L=D+R*p[2];for(let z=0;z<g;z++){let W=y[C+z],U=y[V+z],G=y[P+z],q=y[L+z],H=W+(G-W)*O,j=U+(q-U)*O,K=H+(j-H)*E;b[v++]=K}}}return n.makeTensorInfo([d,c,h,g],"float32",b)}},rf={kernelName:r.ResizeBilinearGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{images:i,dy:o}=t,{alignCorners:u}=s;a([o,i],"resizeBilinearGrad");let l=r.util.computeStrides(i.shape),[p,c,h,d]=i.shape,[,f,m]=o.shape,g=new Float32Array(p*c*h*d),y=[u&&f>1?c-1:c,u&&m>1?h-1:h],b=[u&&f>1?f-1:f,u&&m>1?m-1:m],k=y[0]/b[0],N=y[1]/b[1],v=n.data.get(o.dataId).values,x=0;for(let w=0;w<p;w++){let T=w*l[0];for(let S=0;S<f;S++){let I=S*k,_=Math.floor(I),E=Math.min(Math.ceil(I),c-1),A=T+_*l[1],M=T+E*l[1],D=I-_,$=1-D;for(let F=0;F<m;F++){let B=F*N,O=Math.floor(B),R=Math.min(Math.ceil(B),h-1),C=B-O,V=1-C,P=A+O*l[2],L=A+R*l[2],z=M+O*l[2],W=M+R*l[2],U=$*V,G=$*C,q=D*V,H=D*C;for(let j=0;j<d;j++){let K=v[x++];g[P+j]+=K*U,g[L+j]+=K*G,g[z+j]+=K*q,g[W+j]+=K*H}}}}return n.makeTensorInfo([p,h,c,d],"float32",g)}},rm={kernelName:r.ResizeNearestNeighbor,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{images:i}=t,{alignCorners:o,halfPixelCenters:u,size:l}=s;a(i,"resizeNearestNeighbor");let p=r.util.computeStrides(i.shape),[c,h]=l,[d,f,m,g]=i.shape,y=n.data.get(i.dataId).values,b=new Float32Array(d*c*h*g),k=[o&&c>1?f-1:f,o&&h>1?m-1:m],N=[o&&c>1?c-1:c,o&&h>1?h-1:h],v=k[0]/N[0],x=k[1]/N[1],w=0;for(let T=0;T<d;T++){let S=T*p[0];for(let I=0;I<c;I++){let _=u?v*(I+.5):v*I,E=Math.min(f-1,o?Math.round(_):Math.floor(_));u&&(E=Math.max(0,E));let A=S+E*p[1];for(let M=0;M<h;M++){let D=u?x*(M+.5):x*M,$=Math.min(m-1,o?Math.round(D):Math.floor(D));u&&($=Math.max(0,$));let F=A+$*p[2];for(let B=0;B<g;B++){let O=y[F+B];b[w++]=O}}}}return n.makeTensorInfo([d,c,h,g],i.dtype,b)}},rg={kernelName:r.ResizeNearestNeighborGrad,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{images:i,dy:o}=t,{alignCorners:u}=s;a([o,i],"resizeNearestNeighborGrad");let l=r.util.computeStrides(i.shape),p=r.util.computeStrides(o.shape),[c,h,d,f]=i.shape,[,m,g]=o.shape,y=new Float32Array(c*h*d*f),b=n.data.get(o.dataId).values,k=[u&&m>1?h-1:h,u&&g>1?d-1:d],N=[u&&m>1?m-1:m,u&&g>1?g-1:g],v=k[0]/N[0],x=k[1]/N[1],w=1/v,T=1/x,S=2*Math.ceil(w)+2,I=2*Math.ceil(T)+2;for(let _=0;_<c;_++){let E=_*l[0];for(let A=0;A<h;A++){let M=E+A*l[1],D=Math.floor(A*w),$=Math.floor(D-S/2);for(let F=0;F<d;F++){let B=M+F*l[2],O=Math.floor(F*T),R=Math.floor(O-I/2);for(let C=0;C<f;C++){let V=0;for(let P=0;P<S;P++){let L=P+$;if(L<0||L>=m)continue;let z=E+L*p[1],W=L*v,U=Math.min(h-1,u?Math.round(W):Math.floor(W));if(A===U)for(let G=0;G<I;G++){let q=G+R;if(q<0||q>=g)continue;let H=z+q*p[2],j=q*x,K=Math.min(d-1,u?Math.round(j):Math.floor(j));F===K&&(V+=b[H+C])}}y[B+C]=V}}}}return n.makeTensorInfo(i.shape,i.dtype,y)}},ry={kernelName:r.Reverse,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{dims:o}=s;a(i,"reverse");let u=i.shape.length,l=r.util.parseAxisParam(o,i.shape);if(0===u)return c({inputs:{x:i},backend:n});let p=new r.TensorBuffer(i.shape,i.dtype),h=n.bufferSync(i);for(let d=0;d<p.size;d++){let f=p.indexToLoc(d),m=f.slice();l.forEach(e=>m[e]=i.shape[e]-1-m[e]),p.set(h.get(...m),...f)}return n.makeTensorInfo(p.shape,p.dtype,p.values)}},rb={kernelName:r.RotateWithOffset,backendName:"cpu",kernelFunc({inputs:e,attrs:t,backend:n}){let{image:a}=e,{radians:s,fillValue:i,center:o}=t,u=r.util.getTypedArrayFromDType(a.dtype,r.util.sizeFromShape(a.shape)),[l,p,c,h]=a.shape,[d,f]=r.backend_util.getImageCenter(o,p,c),m=Math.sin(s),g=Math.cos(s),y=n.data.get(a.dataId).values;for(let b=0;b<l;b++){let k=b*c*p*h;for(let N=0;N<p;N++){let v=N*(c*h);for(let x=0;x<c;x++){let w=x*h;for(let T=0;T<h;T++){let S=[l,N,x,T],I=S[2],_=S[1],E=(I-d)*g-(_-f)*m,A=(I-d)*m+(_-f)*g;E=Math.round(E+d),A=Math.round(A+f);let M=i;if("number"!=typeof i&&(M=3===T?255:i[T]),E>=0&&E<c&&A>=0&&A<p){let D=A*(c*h),$=E*h,F=k+D+$+T;M=y[F]}let B=k+v+w+T;u[B]=M}}}}let O=n.write(u,a.shape,a.dtype);return{dataId:O,shape:a.shape,dtype:a.dtype}}},rk=o(r.Round,e=>{let t=Math.floor(e);return e-t<.5?Math.floor(e):e-t>.5?Math.ceil(e):t%2==0?t:t+1}),rN={kernelName:r.Round,backendName:"cpu",kernelFunc:rk},rv=w(e=>1/Math.sqrt(e)),rx=u(r.Rsqrt,rv),rw={kernelName:r.Rsqrt,backendName:"cpu",kernelFunc:rx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function rT(e,t,n,a,s,i,o,u,l,p){let c=e.values,h=t.values;if(0===a)return(0,r.buffer)(n,t.dtype);let d=(0,r.buffer)([a/s,s],t.dtype);"string"==typeof l?d.values.fill(l):"number"==typeof l?d.values.fill(l):"boolean"==typeof l&&d.values.fill(+l);for(let f=0;f<i;f++){let m=[],g=0;for(let y=0;y<o;y++){let b=c[f*o+y];m.push(b),g+=b*u[y]}if(g<0||g>=a/s)throw Error(`Invalid indices: ${m} does not index into ${n}`);for(let k=0;k<s;k++)p?d.values[g*s+k]+=h[f*s+k]:d.values[g*s+k]=0===t.rank?h[0]:h[f*s+k]}return d}let rS={kernelName:r.ScatterNd,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{indices:s,updates:i}=t,{shape:o}=a,{sliceRank:u,numUpdates:l,sliceSize:p,strides:c,outputSize:h}=r.backend_util.calculateShapes(i,s,o),d=n.bufferSync(s),f=n.bufferSync(i),m=rT(d,f,o,h,p,l,u,c,0,!0);return n.makeTensorInfo(o,m.dtype,m.values)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function rI(e,t){let n=0,r=e.length,a=0;for(;n<r;)e[a=Math.floor((n+r)/2)]<t?n=a+1:r=a;return r}function r_(e,t){let n=0,r=e.length,a=0;for(;n<r;)e[a=Math.floor((n+r)/2)]<=t?n=a+1:r=a;return r}let rE={kernelName:r.SearchSorted,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{sortedSequence:s,values:i}=t,{side:o}=a,u=n.data.get(s.dataId).values,l=n.data.get(i.dataId).values,p=function(e,t,n,a,s,i){let o=r.util.getArrayFromDType("int32",n*s);for(let u=0;u<n;++u){let l=e.slice(u*a,(u+1)*a),p=u*s;for(let c=0;c<s;++c)o[p+c]="left"===i?rI(l,t[c+p]):r_(l,t[c+p])}return o}(u,l,s.shape[0],s.shape[1],i.shape[1],o);return n.makeTensorInfo(i.shape,"int32",p)}},rA={kernelName:r.Select,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{condition:s,t:i,e:o}=t;a([s,i,o],"select");let u=s.shape.length,l=n.data.get(s.dataId).values,p=n.data.get(i.dataId).values,c=n.data.get(o.dataId).values,h=(0,r.upcastType)(i.dtype,o.dtype),d=r.util.makeZerosTypedArray(r.util.sizeFromShape(i.shape),h),f=0,m=0===u||u>1||1===i.shape.length?1:r.util.sizeFromShape(i.shape.slice(1));for(let g=0;g<l.length;g++)for(let y=0;y<m;y++)1===l[g]?d[f++]=p[g]:d[f++]=c[g];return n.makeTensorInfo(i.shape,h,d)}},rM=r.backend_util.SELU_SCALEALPHA,rD=r.backend_util.SELU_SCALE,r$=o(r.Selu,e=>e>=0?rD*e:rM*(Math.exp(e)-1)),rF={kernelName:r.Selu,backendName:"cpu",kernelFunc:r$},rB=o(r.Sign,e=>e<0?-1:e>0?1:0),rO={kernelName:r.Sign,backendName:"cpu",kernelFunc:rB},rR=o(r.Sin,e=>Math.sin(e)),rC={kernelName:r.Sin,backendName:"cpu",kernelFunc:rR},rV=o(r.Sinh,e=>Math.sinh(e)),rP={kernelName:r.Sinh,backendName:"cpu",kernelFunc:rV},rL=Math.log(11920928955078125e-23)+2,rz=o(r.Softplus,e=>{let t=Math.exp(e);return e<rL?t:e>-rL?e:Math.log(1+t)}),rW={kernelName:r.Softplus,backendName:"cpu",kernelFunc:rz},rU={kernelName:r.SpaceToBatchND,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{blockShape:o,paddings:u}=s;a([i],"spaceToBatchND");let l=r.util.sizeFromShape(o),p=[[0,0]];p.push(...u);for(let c=1+o.length;c<i.shape.length;++c)p.push([0,0]);let h=n8.kernelFunc({inputs:{x:i},backend:n,attrs:{paddings:p,constantValue:0}}),d=r.backend_util.getReshaped(h.shape,o,l,!1),f=r.backend_util.getPermuted(d.length,o.length,!1),m=r.backend_util.getReshapedPermuted(h.shape,o,l,!1),g=L({inputs:{x:h},backend:n,attrs:{shape:d}}),y=J({inputs:{x:g},backend:n,attrs:{perm:f}}),b=L({inputs:{x:y},backend:n,attrs:{shape:m}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(y),b}},rG={kernelName:r.SparseFillEmptyRows,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{indices:a,values:s,denseShape:i,defaultValue:o}=t;if(1!==i.shape.length)throw Error(`Dense shape must be a vector, saw:
        ${i.shape}`);if(2!==a.shape.length)throw Error(`Indices must be a matrix, saw:
        ${a.shape}`);if(1!==s.shape.length)throw Error(`Values must be a vector, saw:
        ${s.shape}`);if(0!==o.shape.length)throw Error(`Default value must be a scalar, saw:
        ${o.shape}`);let u=n.data.get(a.dataId).values,l=n.data.get(s.dataId).values,p=n.data.get(i.dataId).values,c=n.data.get(o.dataId).values[0],[h,d,f,m,g]=/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a,s,i,o){let u=t[0],l=i[0],p=Array(l),c=Array(u),h=t[1];if(0===l){if(0!==u)throw Error(r.backend_util.getSparseFillEmptyRowsIndicesDenseShapeMismatch(u));let d=r.util.getArrayFromDType(n,0),f=r.util.getArrayFromDType(s,0);return[d,[0,h],f,p,c]}let m=!0,g=0,y=Array(l).fill(0);for(let b=0;b<u;++b){let k=e[b*h];if(k<0)throw Error(r.backend_util.getSparseFillEmptyRowsNegativeIndexErrorMessage(b,k));if(k>=l)throw Error(r.backend_util.getSparseFillEmptyRowsOutOfRangeIndexErrorMessage(b,k,l));++y[k],m=m&&k>=g,g=k}let N=!0;for(let v=0;v<l;++v){let x=0===y[v];p[v]=x,N=N&&!x,y[v]=Math.max(y[v],1),v>0&&(y[v]+=y[v-1])}if(N&&m){for(let w=0;w<u;++w)c[w]=w;return[e,[u,h],a,p,c]}{let T=y[l-1],S=r.util.getArrayFromDType(n,T*h),I=r.util.getArrayFromDType(s,T),_=Array(l).fill(0);for(let E=0;E<u;++E){let A=e[E*h],M=_[A],D=(0===A?0:y[A-1])+M;_[A]++;for(let $=0;$<h;++$)S[D*h+$]=e[E*h+$];I[D]=a[E],c[E]=D}for(let F=0;F<l;++F){let B=_[F];if(0===B){let O=0===F?0:y[F-1];S[O*h+0]=F;for(let R=1;R<h;++R)S[O*h+R]=0;I[O]=o}}return[S,[T,h],I,p,c]}}(u,a.shape,a.dtype,l,s.dtype,p,c);return[n.makeTensorInfo(d,a.dtype,h),n.makeTensorInfo([d[0]],s.dtype,f),n.makeTensorInfo([m.length],"bool",new Uint8Array(m.map(e=>Number(e)))),n.makeTensorInfo([g.length],a.dtype,new Int32Array(g)),]}},rq={kernelName:r.SparseReshape,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{inputIndices:a,inputShape:s,newShape:i}=t;if(2!==a.shape.length)throw Error(`Input indices should be a matrix but received shape
        ${a.shape}`);if(1!==s.shape.length)throw Error(`Input shape should be a vector but received shape
        ${s.shape}`);if(1!==i.shape.length)throw Error(`Target shape should be a vector but received shape ${i.shape}`);let o=Array.from(n.data.get(s.dataId).values),u=n.data.get(a.dataId).values,l=Array.from(n.data.get(i.dataId).values),[p,c,h]=/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a,s){let i=r.util.sizeFromShape(a),o=t[0],u=s.length,l=[],p=1,c=-1;for(let h=0;h<u;++h){let d=s[h];if(-1===d){if(-1!==c)throw Error(r.backend_util.getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(c,h));c=h,l.push(1)}else{if(d<0)throw Error(r.backend_util.getSparseReshapeNegativeOutputDimErrorMessage(h,d));p*=d,l.push(d)}}if(-1!==c){if(p<=0)throw Error(r.backend_util.getSparseReshapeEmptyTensorZeroOutputDimErrorMessage());let f=Math.trunc(i/p);if(p*f!==i)throw Error(r.backend_util.getSparseReshapeInputOutputMultipleErrorMessage(a,l));l[c]=f}let m=r.util.sizeFromShape(l);if(m!==i)throw Error(r.backend_util.getSparseReshapeInputOutputMismatchErrorMessage(a,l));let g=a.length,y=[];if(g>0){y[g-1]=1;for(let b=g-2;b>=0;--b)y[b]=y[b+1]*a[b+1]}let k=[];if(u>0){k[u-1]=1;for(let N=u-2;N>=0;--N)k[N]=k[N+1]*l[N+1]}let v=r.util.getArrayFromDType(n,o*u);for(let x=0;x<o;++x){let w=0;for(let T=0;T<g;++T)w+=e[x*g+T]*y[T];for(let S=0;S<u;++S)v[x*u+S]=Math.trunc(w/k[S]),w%=k[S]}return[v,[o,u],l]}(u,a.shape,a.dtype,o,l);return[n.makeTensorInfo(c,a.dtype,p),n.makeTensorInfo([h.length],i.dtype,new Int32Array(h)),]}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function rH(e,t,n,a,s,i=!1,o=0){let u=a.length,l=[t[0],e.length/t[0]],p=l[1],c=u>0?s[u-1]+1:0;if(c<0)throw Error(r.backend_util.getSparseSegmentReductionNegativeSegmentIdsErrorMessage());let h=t.slice();h[0]=c;let d=h.reduce((e,t)=>e*t,1),f=r.util.getArrayFromDType(n,d);if(0===u)return c>0&&f.fill(o),[f,h];if(c<=0)throw Error(r.backend_util.getSparseSegmentReductionNegativeSegmentIdsErrorMessage());let m=0,g=1,y=0,b=s[m];for(;;){let k=0;if(g<u){if(b===(k=s[g])){++g;continue}if(b>=k)throw Error(r.backend_util.getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage())}if(b<0||b>=c)throw Error(r.backend_util.getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage(b,c));b>y&&f.fill(o,y*p,b*p);for(let N=m;N<g;++N){let v=a[N];if(v<0||v>=l[0])throw Error(r.backend_util.getSparseSegmentReductionIndicesOutOfRangeErrorMessage(N,a[N],l[0]));for(let x=0;x<p;x++)f[b*p+x]+=e[v*p+x]}if(i)for(let w=0;w<p;w++)f[b*p+w]/=g-m;if(m=g,++g,y=b+1,b=k,g>u)break}return y<c&&f.fill(o,y*p,c*p),[f,h]}let rj={kernelName:r.SparseSegmentMean,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{data:r,indices:a,segmentIds:s}=t;if(r.shape.length<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.shape.length)throw Error(`Indices should be a vector but received shape
          ${a.shape}`);if(1!==s.shape.length)throw Error(`Segment ids should be a vector but received shape
          ${s.shape}`);if(a.shape[0]!==s.shape[0])throw Error("segmentIds and indices should have same size.");let i=n.data.get(r.dataId).values,o=n.data.get(a.dataId).values,u=n.data.get(s.dataId).values,[l,p]=rH(i,r.shape,r.dtype,o,u,!0);return n.makeTensorInfo(p,r.dtype,l)}},rK={kernelName:r.SparseSegmentSum,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n}=e,{data:r,indices:a,segmentIds:s}=t;if(r.shape.length<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.shape.length)throw Error(`Indices should be a vector but received shape
         ${a.shape}`);if(1!==s.shape.length)throw Error(`Segment ids should be a vector but received shape
         ${s.shape}`);if(a.shape[0]!==s.shape[0])throw Error("segmentIds and indices should have same size.");let i=n.data.get(r.dataId).values,o=n.data.get(a.dataId).values,u=n.data.get(s.dataId).values,[l,p]=rH(i,r.shape,r.dtype,o,u);return n.makeTensorInfo(p,r.dtype,l)}},rX={kernelName:r.SparseToDense,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{sparseIndices:s,sparseValues:i,defaultValue:o}=t,{outputShape:u}=a,{sliceRank:l,numUpdates:p,sliceSize:c,strides:h,outputSize:d}=r.backend_util.calculateShapes(i,s,u),f=n.bufferSync(s),m;switch(i.dtype){case"bool":{let g=n.bufferSync(i),y=Boolean(n.data.get(o.dataId).values[0]);m=rT(f,g,u,d,c,p,l,h,y,!1);break}case"float32":{let b=n.bufferSync(i),k=n.data.get(o.dataId).values[0];m=rT(f,b,u,d,c,p,l,h,k,!1);break}case"int32":{let N=n.bufferSync(i),v=n.data.get(o.dataId).values[0];m=rT(f,N,u,d,c,p,l,h,v,!1);break}case"string":{let x=n.bufferSync(i),w=r.util.decodeString(n.data.get(o.dataId).values[0]);m=rT(f,x,u,d,c,p,l,h,w,!1);break}default:throw Error(`Unsupported type ${i.dtype}`)}return n.makeTensorInfo(u,m.dtype,m.values)}},rZ={kernelName:r.SplitV,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{x:s}=t,{numOrSizeSplits:i,axis:o}=a,u=r.util.parseAxisParam(o,s.shape)[0],l=r.backend_util.prepareSplitSize(s,i,u),p=Array(s.shape.length).fill(0),c=s.shape.slice();return l.map(e=>{let t=[...c];t[u]=e;let r=eT({inputs:{x:s},backend:n,attrs:{begin:p,size:t}});return p[u]+=e,r})}};w(e=>Math.sqrt(e));let rQ=o(r.Sqrt,e=>Math.sqrt(e)),rY={kernelName:r.Sqrt,backendName:"cpu",kernelFunc:rQ},rJ={kernelName:r.Square,backendName:"cpu",kernelFunc({inputs:e,backend:t}){let{x:n}=e;a(n,"square");let r=t.data.get(n.dataId).values,s=new Float32Array(r.length);for(let i=0;i<r.length;++i){let o=r[i];s[i]=o*o}let u=t.write(s,n.shape,n.dtype);return{dataId:u,shape:n.shape,dtype:n.dtype}}},r0=m((e,t)=>{let n=e-t;return n*n}),r1=B(r.SquaredDifference,r0),r2={kernelName:r.SquaredDifference,backendName:"cpu",kernelFunc:r1},r3=o(r.Step,(e,t)=>isNaN(e)?NaN:e>0?1:t.alpha),r6={kernelName:r.Step,backendName:"cpu",kernelFunc:r3},r4={kernelName:r.StridedSlice,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{begin:o,end:u,strides:l,beginMask:p,endMask:c,ellipsisMask:h,newAxisMask:d,shrinkAxisMask:f}=s;a(i,"stridedSlice");let{finalShapeSparse:m,finalShape:g,isIdentity:y,sliceDim0:b,isSimpleSlice:k,begin:N,end:v,strides:x}=r.slice_util.sliceInfo(i.shape,o,u,l,p,c,h,d,f),w;if(y)w=L({inputs:{x:i},backend:n,attrs:{shape:g}});else if(b||k){r.util.assert(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let T=r.slice_util.computeOutShape(N,v,x),S=eT({inputs:{x:i},backend:n,attrs:{begin:N,size:T}});w=L({inputs:{x:S},backend:n,attrs:{shape:g}}),n.disposeIntermediateTensorInfo(S)}else{let I=n.bufferSync(i),_=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a){let s=(0,r.buffer)(e,t.dtype);for(let i=0;i<s.size;i++){let o=s.indexToLoc(i),u=Array(o.length);for(let l=0;l<u.length;l++)u[l]=o[l]*n[l]+a[l];s.set(t.get(...u),...o)}return s}(m,I,x,N);w=n.makeTensorInfo(g,_.dtype,_.values)}return w}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class r5{constructor(e,t,n,a,s,i){this.separator=r.util.encodeString(e),this.nGramWidths=t,this.leftPad=r.util.encodeString(n),this.rightPad=r.util.encodeString(a),this.padWidth=s,this.preserveShort=i}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){let n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,a,s){for(let i=0;i<a;++i){let o=this.getPadWidth(s),u=Math.max(0,o-i),l=Math.max(0,o-(a-(i+1))),p=s-(u+l),c=t+(u>0?0:i-o),h=0;h+=u*this.leftPad.length;for(let d=0;d<p;++d)h+=e[c+d].length;h+=l*this.rightPad.length;let f=u+l+p-1;h+=f*this.separator.length,n[r+i]=new Uint8Array(h);let m=n[r+i],g=0,y=e=>e.forEach(e=>m[g++]=e);for(let b=0;b<u;++b)y(this.leftPad),y(this.separator);for(let k=0;k<p-1;++k)y(e[c+k]),y(this.separator);if(p>0){y(e[c+p-1]);for(let N=0;N<l;++N)y(this.separator),y(this.rightPad)}else{for(let v=0;v<l-1;++v)y(this.rightPad),y(this.separator);y(this.rightPad)}}}compute(e,t){let n=e.length,a=t.length;if(a>0){let s=t[0];if(0!==s)throw Error(`First split value must be 0, got ${s}`);for(let i=1;i<a;++i){let o=t[i]>=s;if(!(o=o&&t[i]<=n))throw Error(`Invalid split value ${t[i]}, must be in [${s}, ${n}]`);s=t[i]}if(s!==n)throw Error(`Last split value must be data size. Expected ${n}, got ${s}`)}let u=a-1,l=r.util.getArrayFromDType("int32",a);if(0===n||0===a){let p=Array(n);for(let c=0;c<=u;++c)l[c]=0;return[p,l]}l[0]=0;for(let h=1;h<=u;++h){let d=t[h]-t[h-1],f=0;this.nGramWidths.forEach(e=>{f+=this.getNumNGrams(d,e)}),this.preserveShort&&d>0&&0===f&&(f=1),l[h]=l[h-1]+f}let m=Array(l[u]);for(let g=0;g<u;++g){let y=t[g],b=l[g];if(this.nGramWidths.forEach(n=>{let r=t[g+1]-t[g],a=this.getNumNGrams(r,n);this.createNGrams(e,y,m,b,a,n),b+=a}),this.preserveShort&&b===l[g]){let k=t[g+1]-t[g];if(0===k)continue;let N=k+2*this.padWidth;this.createNGrams(e,y,m,b,1,N)}}return[m,l]}}let r8={kernelName:r.StringNGrams,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){var t,n,r,a,s,i,o,u;let{inputs:l,backend:p,attrs:c}=e,{separator:h,nGramWidths:d,leftPad:f,rightPad:m,padWidth:g,preserveShortSequences:y}=c,{data:b,dataSplits:k}=l,N=p.data.get(b.dataId).values,v=p.data.get(k.dataId).values,[x,w]=new r5(h,d,f,m,g,y).compute(N,v);return[p.makeTensorInfo([x.length],"string",x),p.makeTensorInfo(k.shape,"int32",w),]}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function r7(e,t,n,r){if(!e.length)return;if(0===t.length){for(let a=0;a<e.length;++a)r.push(e.subarray(a,a+1));return}if(1===t.length){let s=t[0],i=e.indexOf(s);for(;-1!==i;){let o=e.subarray(0,i);n&&0===o.length||r.push(o),i=(e=e.subarray(i+1)).indexOf(s)}n&&0===e.length||r.push(e);return}let u=0;for(let l=0;l<e.length+1;l++)if(l===e.length||-1!==t.indexOf(e[l])){let p=e.subarray(u,l);n&&0===p.length||r.push(p),u=l+1}}let r9={kernelName:r.StringSplit,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{skipEmpty:s}=a,{input:i,delimiter:o}=t;if("string"!==i.dtype)throw Error("Input must be of datatype string");if(1!==i.shape.length)throw Error(`Input must be a vector, got shape: ${i.shape}`);if(0!==o.shape.length)throw Error(`Delimiter must be a scalar, got shape: ${o.shape}`);let u=n.data.get(i.dataId).values,l=n.data.get(o.dataId).values[0],[p,c,h]=function(e,t,n){let a=e.length,s=[],i=0,o=0,u=Array(a);for(let l=0;l<a;++l){let p=s.length;r7(e[l],t,n,s);let c=s.length-p;u[l]=c,i+=c,o=Math.max(o,c)}let h=r.util.getArrayFromDType("int32",2*i),d=Array(i),f=[a,o],m=0;for(let g=0;g<a;++g)for(let y=0;y<u[g];++y)h[2*m]=g,h[2*m+1]=y,d[m]=s[m],++m;return[h,d,f]}(u,l,s),d=c.length;return[n.makeTensorInfo([d,2],"int32",p),n.makeTensorInfo([d],"string",c),n.makeTensorInfo([2],"int32",new Int32Array(h))]}},ae={kernelName:r.StringToHashBucketFast,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:a}=e,{numBuckets:s}=a,{input:i}=t;if("string"!==i.dtype)throw Error("Input must be of datatype string");if(s<=0)throw Error("Number of buckets must be at least 1");let o=n.data.get(i.dataId).values,u=/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=r.util.getArrayFromDType("int32",e.length);for(let a=0;a<e.length;++a)n[a]=r.util.fingerPrint64(e[a]).modulo(t).getLowBitsUnsigned();return n}(o,s);return n.makeTensorInfo(i.shape,"int32",u)}},at=o(r.Tan,e=>Math.tan(e)),an={kernelName:r.Tan,backendName:"cpu",kernelFunc:at},ar=o(r.Tanh,e=>Math.tanh(e)),aa={kernelName:r.Tanh,backendName:"cpu",kernelFunc:ar},as={kernelName:r.Tile,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{reps:o}=s;a(i,"tile");let u=/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=Array(e.rank);for(let a=0;a<n.length;a++)n[a]=e.shape[a]*t[a];let s=(0,r.buffer)(n,e.dtype);for(let i=0;i<s.values.length;++i){let o=s.indexToLoc(i),u=Array(e.rank);for(let l=0;l<u.length;l++)u[l]=o[l]%e.shape[l];let p=e.locToIndex(u);s.values[i]=e.values[p]}return s}(n.bufferSync(i),o);return n.makeTensorInfo(u.shape,u.dtype,u.values)}},ai=(e,t)=>{let n=t.value-e.value;return 0===n?e.index-t.index:n};function ao(e,t,n=0,a=e.length-1){for(;a>n;){if(a-n>600){let s=a-n+1,i=t-n+1,o=Math.log(s),u=.5*Math.exp(2*o/3),l=.5*Math.sqrt(o*u*(s-u)/s)*Math.sign(i-s/2),p=Math.max(n,Math.floor(t-i*u/s+l)),c=Math.min(a,Math.floor(t+(s-i)*u/s+l));ao(e,t,p,c)}let h=e[t],d=n,f=a;for(r.util.swap(e,n,t),ai(e[a],h)>0&&r.util.swap(e,n,a);d<f;){for(r.util.swap(e,d,f),d++,f--;0>ai(e[d],h);)d+=1;for(;ai(e[f],h)>0;)f-=1}0===ai(e[n],h)?r.util.swap(e,n,f):(f+=1,r.util.swap(e,f,a)),f<=t&&(n=f+1),t<=f&&(a=f-1)}}let au={kernelName:r.TopK,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i}=t,{k:o,sorted:u}=s;a(i,"topk");let l=n.data.get(i.dataId).values,[p,c]=function(e,t,n,a,s){let i=t[t.length-1],[o,u]=[e.length/i,i],l=r.util.getTypedArrayFromDType(n,o*a),p=r.util.getTypedArrayFromDType("int32",o*a);for(let c=0;c<o;c++){let h=c*u,d=e.subarray(h,h+u),f=Array(d.length);d.forEach((e,t)=>f[t]={value:e,index:t}),a<f.length&&(ao(f,a),f=f.slice(0,a)),s&&f.sort(ai);let m=c*a,g=l.subarray(m,m+a),y=p.subarray(m,m+a);for(let b=0;b<a;b++)g[b]=f[b].value,y[b]=f[b].index}let k=t.slice();return k[k.length-1]=a,[(0,r.buffer)(k,n,l),(0,r.buffer)(k,"int32",p)]}(l,i.shape,i.dtype,o,u);return[n.makeTensorInfo(p.shape,p.dtype,p.values),n.makeTensorInfo(c.shape,c.dtype,c.values)]}},al={kernelName:r.Transform,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,attrs:n,backend:a}=e,{image:s,transforms:i}=t,{interpolation:o,fillMode:u,fillValue:l,outputShape:p}=n,[c,h,d,f]=s.shape,[m,g]=null!=p?p:[h,d],y=[c,m,g,f],b=r.util.computeStrides(s.shape),k=b[0],N=b[1],v=b[2],x=r.util.computeStrides(y),w=x[0],T=x[1],S=x[2],I=r.util.getTypedArrayFromDType(s.dtype,r.util.sizeFromShape(y));I.fill(l);let _=a.data.get(s.dataId).values,E=a.data.get(i.dataId).values;for(let A=0;A<c;++A){let M=1===i.shape[0]?E:E.subarray(8*A,8*A+8);for(let D=0;D<m;++D)for(let $=0;$<g;++$)for(let F=0;F<f;++F){let B,O=M[6]*$+M[7]*D+1;if(0===O)continue;let R=(M[0]*$+M[1]*D+M[2])/O,C=(M[3]*$+M[4]*D+M[5])/O,V=ap(R,d,u),P=ap(C,h,u);switch(o){case"nearest":B=ah(_,h,d,k,N,v,A,P,V,F,l);break;case"bilinear":B=ad(_,h,d,k,N,v,A,P,V,F,l);break;default:throw Error(`Error in Transform: Expect 'nearest' or 'bilinear', but got ${o}`)}let L=A*w+D*T+$*S+F;I[L]=B}return a.makeTensorInfo(y,s.dtype,I)}let z=a.write(I,y,s.dtype);return{dataId:z,shape:s.shape,dtype:s.dtype}}};function ap(e,t,n){var a,s,i,o,u,l;switch(n){case"reflect":return function(e,t){let n=e;if(n<0){if(t<=1)n=0;else{let a=2*t;n<a&&(n=a*Math.trunc(-n/a)+n),n=n<-t?n+a:-n-1}}else if(n>t-1){if(t<=1)n=0;else{let s=2*t;(n-=s*Math.trunc(n/s))>=t&&(n=s-n-1)}}return r.util.clamp(0,n,t-1)}(e,t);case"wrap":let p;return a=e,s=t,p=a,p<0?s<=1?p=0:p+=s*(Math.trunc(-p/(s-1))+1):p>s-1&&(s<=1?p=0:p-=s*Math.trunc(p/(s-1))),r.util.clamp(0,p,s-1);case"nearest":return i=e,o=t,r.util.clamp(0,i,o-1);default:return u=e,l=t,u}}function ac(e,t,n,r,a,s,i,o,u,l,p){return 0<=o&&o<t&&0<=u&&u<n?e[i*r+o*a+u*s+l]:p}function ah(e,t,n,r,a,s,i,o,u,l,p){let c=Math.round(o),h=Math.round(u);return ac(e,t,n,r,a,s,i,c,h,l,p)}function ad(e,t,n,r,a,s,i,o,u,l,p){let c=Math.floor(o),h=Math.floor(u),d=c+1,f=h+1,m=(f-u)*ac(e,t,n,r,a,s,i,c,h,l,p)+(u-h)*ac(e,t,n,r,a,s,i,c,f,l,p),g=(f-u)*ac(e,t,n,r,a,s,i,d,h,l,p)+(u-h)*ac(e,t,n,r,a,s,i,d,f,l,p);return(d-o)*m+(o-c)*g}let af={kernelName:r.Unique,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,attrs:n,backend:s}=e,{axis:i}=n,{x:o}=t;a(o,"unique");let u=s.data.get(o.dataId).values,{outputValues:l,outputShape:p,indices:c}=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,a){let s=r.util.parseAxisParam(t,n)[0],i=[1,n[0],1];for(let o=0;o<s;o++)i[0]*=n[o];i[1]=n[s];for(let u=s+1;u<n.length;u++)i[2]*=n[u];let l={},p=new Int32Array(n[s]),c=new r.TensorBuffer(i,a,e),h=[],d=1===i[0]&&1===i[2];for(let f=0;f<n[s];f++){let m;if(d)m=e[f].toString();else{let g=[];for(let y=0;y<i[0];y++)for(let b=0;b<i[2];b++)g.push(c.get(y,f,b));m=g.join(",")}if(void 0!==l[m])p[f]=l[m];else{let k=Object.keys(l).length;l[m]=k,p[f]=k,h.push(f)}}let N=i.slice();N[1]=Object.keys(l).length;let v=new r.TensorBuffer(N,a);h.forEach((e,t)=>{for(let n=0;n<i[0];n++)for(let r=0;r<i[2];r++)v.set(c.get(n,e,r),n,t,r)});let x=n.slice();return x[s]=N[1],{outputValues:v.values,outputShape:x,indices:p}}(u,i,o.shape,o.dtype);return[s.makeTensorInfo(p,o.dtype,l),s.makeTensorInfo([c.length],"int32",c),]}},am={kernelName:r.Unpack,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:r}=e,{value:a}=t,{axis:s}=r;s<0&&(s+=a.shape.length);let i=a.shape.length,o=a.shape[s],u=Array(i-1),l=0;for(let p=0;p<i;p++)p!==s&&(u[l++]=a.shape[p]);let c=Array(i).fill(0),h=a.shape.slice();h[s]=1;let d=Array(o);for(let f=0;f<d.length;f++){c[s]=f;let m=eT({inputs:{x:a},backend:n,attrs:{begin:c,size:h}});d[f]=L({inputs:{x:m},backend:n,attrs:{shape:u}}),n.disposeIntermediateTensorInfo(m)}return d}},ag={kernelName:r.UnsortedSegmentSum,backendName:"cpu",kernelFunc:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let{inputs:t,backend:n,attrs:s}=e,{x:i,segmentIds:o}=t,{numSegments:u}=s;a(i,"unsortedSegmentSum");let l=i.shape.length,p=o.shape.length,c=[],h=[],d=l-p,f=o;for(let m=0;m<d;++m){let g=tw({inputs:{input:f},backend:n,attrs:{dim:m+1}});f=g,h.push(g)}for(let y=0;y<u;++y){let b=r.util.createScalarValue(y,"int32"),k=n.makeTensorInfo([],"int32",b),N=tp({inputs:{a:k,b:f},backend:n}),v=$({inputs:{x:N},backend:n,attrs:{dtype:"float32"}}),x=tr({inputs:{a:v,b:i},backend:n}),w=ts({inputs:{x:x},backend:n,attrs:{axis:0,keepDims:!1}});c.push(w),h.push(k),h.push(N),h.push(v),h.push(x),h.push(w)}let T=n4({inputs:c,backend:n,attrs:{axis:0}});return h.forEach(e=>n.disposeIntermediateTensorInfo(e)),T}};for(let ay of[G,H,K,Z,P,Q,et,en,er,ea,ei,eu,ep,ed,em,ek,eN,ev,ex,U,ew,eI,eE,eA,F,e$,eB,E,eR,eL,eW,eU,eG,eq,eH,ej,eX,eQ,eY,eJ,e0,e1,e2,e6,e4,e5,e8,e7,e9,te,to,p,tu,tc,tk,tx,tT,t_,tC,tP,tL,tU,tH,tj,tK,tX,tZ,tJ,t2,h,t3,eV,t4,t8,t9,f,nn,ns,ni,nl,nc,nf,ng,nk,nN,nv,nw,nI,n_,nE,nA,nM,nD,n$,nF,nR,nC,nL,nG,ta,nq,nj,nX,nQ,n0,n1,n6,n5,n8,re,b,rt,rr,ra,rl,rp,D,tM,rh,N,x,z,rd,rf,rm,rg,ry,rb,rN,rw,rS,rE,rA,rF,S,rO,rC,rP,eS,nU,rW,rU,rG,rq,rj,rK,rX,rZ,rY,rJ,r2,r6,r4,r8,r9,ae,tB,ti,an,aa,as,au,al,ee,af,am,ag,n3])(0,r.registerKernel)(ay);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ },7921:function(e,t,n){"use strict";n.r(t),n.d(t,{GraphModel:function(){return e9},deregisterOp:function(){return $},loadGraphModel:function(){return te},loadGraphModelSync:function(){return tt},registerOp:function(){return D},version_converter:function(){return tn}});var r,a,s,i,o,u={};n.r(u),n.d(u,{json:function(){return L}});var l={};n.r(l),n.d(l,{json:function(){return z}});var p={};n.r(p),n.d(p,{json:function(){return W}});var c={};n.r(c),n.d(c,{json:function(){return U}});var h={};n.r(h),n.d(h,{json:function(){return G}});var d={};n.r(d),n.d(d,{json:function(){return q}});var f={};n.r(f),n.d(f,{json:function(){return H}});var m={};n.r(m),n.d(m,{json:function(){return j}});var g={};n.r(g),n.d(g,{json:function(){return K}});var y={};n.r(y),n.d(y,{json:function(){return X}});var b={};n.r(b),n.d(b,{json:function(){return Z}});var k={};n.r(k),n.d(k,{json:function(){return Q}});var N={};n.r(N),n.d(N,{json:function(){return Y}});var v={};n.r(v),n.d(v,{json:function(){return J}});var x={};n.r(x),n.d(x,{json:function(){return ee}});var w={};n.r(w),n.d(w,{json:function(){return et}});var T={};n.r(T),n.d(T,{json:function(){return en}});var S={};n.r(S),n.d(S,{json:function(){return er}});var I={};n.r(I),n.d(I,{json:function(){return ea}});var _={};n.r(_),n.d(_,{OP_SCOPE_SUFFIX:function(){return ex.zvA},abs:function(){return ex.WnP},acos:function(){return ex.Khb},acosh:function(){return ex.__u},add:function(){return ex.IHx},addN:function(){return ex.QBD},all:function(){return ex.$6P},any:function(){return ex.YjB},argMax:function(){return ex.NqF},argMin:function(){return ex.vHJ},asin:function(){return ex.ZRM},asinh:function(){return ex.VfV},atan:function(){return ex.z4N},atan2:function(){return ex.fvJ},atanh:function(){return ex.C80},avgPool:function(){return ex.wS1},avgPool3d:function(){return ex.uR5},basicLSTMCell:function(){return ex.zEQ},batchNorm:function(){return ex.tgs},batchNorm2d:function(){return ex.Dxk},batchNorm3d:function(){return ex.JY5},batchNorm4d:function(){return ex.p3b},batchToSpaceND:function(){return ex.E4h},bincount:function(){return ex.yE8},booleanMaskAsync:function(){return ex.anm},broadcastArgs:function(){return ex.XsQ},broadcastTo:function(){return ex.UFq},buffer:function(){return ex.f3b},cast:function(){return ex.pju},ceil:function(){return ex.mDi},clipByValue:function(){return ex.iUl},clone:function(){return ex.d9v},complex:function(){return ex.PYB},concat:function(){return ex.zoF},concat1d:function(){return ex.gME},concat2d:function(){return ex.Izb},concat3d:function(){return ex.MNy},concat4d:function(){return ex.ZaL},conv1d:function(){return ex.PAt},conv2d:function(){return ex.Tek},conv2dTranspose:function(){return ex.bc},conv3d:function(){return ex.pdZ},conv3dTranspose:function(){return ex.$QV},cos:function(){return ex.mCk},cosh:function(){return ex.f9Y},cosineWindow:function(){return ex.mew},cumprod:function(){return ex.$Gn},cumsum:function(){return ex.zbp},denseBincount:function(){return ex.ppE},depthToSpace:function(){return ex.nTT},depthwiseConv2d:function(){return ex.B10},diag:function(){return ex.Ka3},dilation2d:function(){return ex.WmZ},div:function(){return ex.hiC},divNoNan:function(){return ex.NTj},dot:function(){return ex.AKD},dropout:function(){return ex.rvX},einsum:function(){return ex.WYO},elu:function(){return ex.pyx},enclosingPowerOfTwo:function(){return ex.GRh},equal:function(){return ex.DgJ},erf:function(){return ex.qNN},euclideanNorm:function(){return ex.d2q},exp:function(){return ex.Qqt},expandDims:function(){return ex.dt4},expm1:function(){return ex.t$B},eye:function(){return ex.iyy},fft:function(){return ex.kp_},fill:function(){return ex.hlL},floor:function(){return ex.GWj},floorDiv:function(){return ex.qPi},fused:function(){return ex.imm},gather:function(){return ex.Iqj},gatherND:function(){return ex.dbB},greater:function(){return ex.pjt},greaterEqual:function(){return ex.brS},ifft:function(){return ex.Sxn},imag:function(){return ex.asL},image:function(){return ex.BHj},inTopKAsync:function(){return ex.V3u},irfft:function(){return ex.wx0},isFinite:function(){return ex.xVT},isInf:function(){return ex.UWc},isNaN:function(){return ex.i2d},leakyRelu:function(){return ex.hi7},less:function(){return ex.d9m},lessEqual:function(){return ex.zN1},linalg:function(){return ex.$r2},linspace:function(){return ex.SX3},localResponseNormalization:function(){return ex.G9k},log:function(){return ex.cM7},log1p:function(){return ex.Krr},logSigmoid:function(){return ex.e_t},logSoftmax:function(){return ex.CmS},logSumExp:function(){return ex.l_t},logicalAnd:function(){return ex.HvI},logicalNot:function(){return ex.hJK},logicalOr:function(){return ex.K5V},logicalXor:function(){return ex.egP},losses:function(){return ex.MB5},lowerBound:function(){return ex.eab},matMul:function(){return ex.OI3},max:function(){return ex.Fp7},maxPool:function(){return ex._sB},maxPool3d:function(){return ex.YQQ},maxPoolWithArgmax:function(){return ex.Ip$},maximum:function(){return ex.gWQ},mean:function(){return ex.J69},meshgrid:function(){return ex.ry_},min:function(){return ex.VV$},minimum:function(){return ex.LTh},mirrorPad:function(){return ex.VdP},mod:function(){return ex.wQq},moments:function(){return ex.Gi7},movingAverage:function(){return ex.p_},mul:function(){return ex.dC7},multiRNNCell:function(){return ex.rq4},multinomial:function(){return ex.SJ_},neg:function(){return ex.W76},norm:function(){return ex.KOy},notEqual:function(){return ex.Quu},oneHot:function(){return ex.lfX},ones:function(){return ex.iUs},onesLike:function(){return ex.JpU},op:function(){return ex.op},outerProduct:function(){return ex.N2O},pad:function(){return ex.vku},pad1d:function(){return ex.pNR},pad2d:function(){return ex.koy},pad3d:function(){return ex.t1L},pad4d:function(){return ex.lGY},pool:function(){return ex.d_R},pow:function(){return ex.sQ3},prelu:function(){return ex.AL3},print:function(){return ex.S0v},prod:function(){return ex.WVs},raggedGather:function(){return ex.$gW},raggedRange:function(){return ex.VT$},raggedTensorToTensor:function(){return ex.N89},rand:function(){return ex.TN_},randomGamma:function(){return ex.wzB},randomNormal:function(){return ex.nGf},randomStandardNormal:function(){return ex.ruB},randomUniform:function(){return ex.LGj},range:function(){return ex.w6H},real:function(){return ex.kwC},reciprocal:function(){return ex.M25},relu:function(){return ex.UYe},relu6:function(){return ex.btT},reshape:function(){return ex.XLQ},reverse:function(){return ex.GYS},reverse1d:function(){return ex.SDf},reverse2d:function(){return ex.diP},reverse3d:function(){return ex.sx7},reverse4d:function(){return ex.mG2},rfft:function(){return ex.QEs},round:function(){return ex.NMM},rsqrt:function(){return ex.bp0},scalar:function(){return ex.iD$},scatterND:function(){return ex.snQ},searchSorted:function(){return ex.zcT},selu:function(){return ex.U8D},separableConv2d:function(){return ex.U_I},setdiff1dAsync:function(){return ex.ODp},sigmoid:function(){return ex.XD2},sign:function(){return ex.Xxe},signal:function(){return ex.tdS},sin:function(){return ex.O$l},sinh:function(){return ex.R_K},slice:function(){return ex.tPi},slice1d:function(){return ex.jZU},slice2d:function(){return ex.SmN},slice3d:function(){return ex.CnO},slice4d:function(){return ex.p0P},softmax:function(){return ex.XAC},softplus:function(){return ex.Wvh},spaceToBatchND:function(){return ex.fBT},sparse:function(){return ex.rVs},sparseToDense:function(){return ex.ers},spectral:function(){return ex.uN7},split:function(){return ex.Vl2},sqrt:function(){return ex._b3},square:function(){return ex.h62},squaredDifference:function(){return ex.$i},squeeze:function(){return ex.L9e},stack:function(){return ex.knu},step:function(){return ex.Nbs},stridedSlice:function(){return ex.NXj},string:function(){return ex.Z_8},sub:function(){return ex.luU},sum:function(){return ex.Smz},tan:function(){return ex.ORZ},tanh:function(){return ex.AEp},tensor:function(){return ex.XeE},tensor1d:function(){return ex.RRF},tensor2d:function(){return ex.odF},tensor3d:function(){return ex.wOQ},tensor4d:function(){return ex.yXz},tensor5d:function(){return ex.Bfx},tensor6d:function(){return ex.xZs},tile:function(){return ex.Gg6},topk:function(){return ex.hg7},transpose:function(){return ex.p4s},truncatedNormal:function(){return ex.Xu6},unique:function(){return ex.Two},unsortedSegmentSum:function(){return ex.pUJ},unstack:function(){return ex.HHK},upperBound:function(){return ex.GaM},variable:function(){return ex.VD$},where:function(){return ex.arb},whereAsync:function(){return ex.itS},zeros:function(){return ex.lls},zerosLike:function(){return ex.P84}});var E=n(5793);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let A=(0,E.env)();A.registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,e=>{e&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")}),(r=i||(i={}))[r.DT_INVALID=0]="DT_INVALID",r[r.DT_FLOAT=1]="DT_FLOAT",r[r.DT_DOUBLE=2]="DT_DOUBLE",r[r.DT_INT32=3]="DT_INT32",r[r.DT_UINT8=4]="DT_UINT8",r[r.DT_INT16=5]="DT_INT16",r[r.DT_INT8=6]="DT_INT8",r[r.DT_STRING=7]="DT_STRING",r[r.DT_COMPLEX64=8]="DT_COMPLEX64",r[r.DT_INT64=9]="DT_INT64",r[r.DT_BOOL=10]="DT_BOOL",r[r.DT_QINT8=11]="DT_QINT8",r[r.DT_QUINT8=12]="DT_QUINT8",r[r.DT_QINT32=13]="DT_QINT32",r[r.DT_BFLOAT16=14]="DT_BFLOAT16",r[r.DT_QINT16=15]="DT_QINT16",r[r.DT_QUINT16=16]="DT_QUINT16",r[r.DT_UINT16=17]="DT_UINT16",r[r.DT_COMPLEX128=18]="DT_COMPLEX128",r[r.DT_HALF=19]="DT_HALF",r[r.DT_RESOURCE=20]="DT_RESOURCE",r[r.DT_VARIANT=21]="DT_VARIANT",r[r.DT_UINT32=22]="DT_UINT32",r[r.DT_UINT64=23]="DT_UINT64",r[r.DT_FLOAT_REF=101]="DT_FLOAT_REF",r[r.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",r[r.DT_INT32_REF=103]="DT_INT32_REF",r[r.DT_UINT8_REF=104]="DT_UINT8_REF",r[r.DT_INT16_REF=105]="DT_INT16_REF",r[r.DT_INT8_REF=106]="DT_INT8_REF",r[r.DT_STRING_REF=107]="DT_STRING_REF",r[r.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",r[r.DT_INT64_REF=109]="DT_INT64_REF",r[r.DT_BOOL_REF=110]="DT_BOOL_REF",r[r.DT_QINT8_REF=111]="DT_QINT8_REF",r[r.DT_QUINT8_REF=112]="DT_QUINT8_REF",r[r.DT_QINT32_REF=113]="DT_QINT32_REF",r[r.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",r[r.DT_QINT16_REF=115]="DT_QINT16_REF",r[r.DT_QUINT16_REF=116]="DT_QUINT16_REF",r[r.DT_UINT16_REF=117]="DT_UINT16_REF",r[r.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",r[r.DT_HALF_REF=119]="DT_HALF_REF",r[r.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",r[r.DT_VARIANT_REF=121]="DT_VARIANT_REF",r[r.DT_UINT32_REF=122]="DT_UINT32_REF",r[r.DT_UINT64_REF=123]="DT_UINT64_REF",(s=(a=o||(o={})).CheckpointFormatVersion||(a.CheckpointFormatVersion={}))[s.LEGACY=0]="LEGACY",s[s.V1=1]="V1",s[s.V2=2]="V2";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let M={};function D(e,t){M[e]={tfOpName:e,category:"custom",inputs:[],attrs:[],customExecutor:t}}function $(e){delete M[e]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function F(e,t,n,r,a){let s=t.inputParams[e];if(s&&void 0!==s.inputIndexStart){let i=s.inputIndexStart,o=0===s.inputIndexEnd?void 0:void 0===s.inputIndexEnd?i+1:s.inputIndexEnd;if("tensor"===s.type)return B(t.inputNames[s.inputIndexStart],n,r,a);if("tensors"===s.type){let u=t.inputNames.slice(i,o);return u.map(e=>B(e,n,r,a))}let l=B(t.inputNames.slice(i)[0],n,r,a),p=l.dataSync();return"number"===s.type?p[0]:E.util.toNestedArray(l.shape,p)}let c=t.attrParams[e];return c&&c.value}function B(e,t,n,r){let[a,s]=C(e);if(null!=r){let i=r.getHashTableHandleByName(a);if(null!=i)return i}let o=n.currentContextIds.find(e=>!!t[R(a,e)]);return void 0!==o?t[R(a,o)][s]:void 0}function O(e,t){let[n,r,a]=C(e);return[R(n,t&&t.currentContextId),r,a]}function R(e,t){return t?`${e}-${t}`:e}function C(e){let t=e.split(":");if(1===t.length)return[e,0,void 0];let n=t[0],r=3===t.length?t[1]:void 0,a=Number(t[t.length-1]);return[n,a,r]}function V(e,t,n){let r=F("pad",e,t,n);if("explicit"===r){r=F("explicitPaddings",e,t,n);let a=[[0,0],[0,0],[0,0],[0,0]];for(let s=0;s<4;s++)a[s][0]=r[2*s],a[s][1]=r[2*s+1];return a}return r}function P(e){return e.kept?e:(0,E.clone)(e)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let L=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],z=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Prod",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axes",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],W=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],U=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],G=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],q=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],H=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],j=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],K=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]}],X=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Z=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Q=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]}],Y=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"SparseToDense",category:"normalization",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!0,notSupported:!0}]}],J=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],ee=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]}],et=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],en=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],er=[{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],ea=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}];var es=n(1876).Buffer;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class ei{static get Instance(){return this._instance||(this._instance=new this)}constructor(){let e=[].concat(...[u,l,p,c,h,d,f,m,g,y,b,k,N,v,x,w,T,S,I].map(e=>e.json));this.opMappers=e.reduce((e,t)=>(e[t.tfOpName]=t,e),{})}transformGraph(e,t={}){let n=e.node,r=[],a=[],s=[],i=n.reduce((e,t)=>(e[t.name]=this.mapNode(t),t.op.startsWith("Placeholder")?r.push(e[t.name]):"Const"===t.op?a.push(e[t.name]):(null==t.input||0===t.input.length)&&s.push(e[t.name]),e),{}),o=[],u=[],l={},p={};null!=t&&(l=this.mapSignatureEntries(t.inputs),p=this.mapSignatureEntries(t.outputs));let c=Object.keys(i);c.forEach(e=>{let t=i[e];t.inputNames.forEach((e,n)=>{let[r,,a]=O(e),s=i[r];if(null!=s.outputs){let o=s.outputs.indexOf(a);if(-1!==o){let u=`${r}:${o}`;t.inputNames[n]=u}}t.inputs.push(s),s.children.push(t)})}),0===Object.keys(p).length?c.forEach(e=>{let t=i[e];0===t.children.length&&u.push(t)}):Object.keys(p).forEach(e=>{let[t,]=O(e),n=i[t];null!=n&&(n.signatureKey=p[e],u.push(n))}),Object.keys(l).length>0?Object.keys(l).forEach(e=>{let[t,]=O(e),n=i[t];n&&(n.signatureKey=l[e],o.push(n))}):o=r;let h={};null!=e.library&&null!=e.library.function&&(h=e.library.function.reduce((e,t)=>(e[t.signature.name]=this.mapFunction(t),e),{}));let d={nodes:i,inputs:o,outputs:u,weights:a,placeholders:r,signature:t,functions:h};return s.length>0&&(d.initNodes=s),d}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,n)=>(t[e[n].name]=n,t),{})}mapNode(e){var t;let n=M[t=e.op]||this.opMappers[e.op]||{};null==e.attr&&(e.attr={});let r={name:e.name,op:e.op,category:n.category,inputNames:(e.input||[]).map(e=>e.startsWith("^")?e.slice(1):e),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:n.outputs};return null!=n.inputs&&(r.inputParams=n.inputs.reduce((e,t)=>(e[t.name]={type:t.type,inputIndexStart:t.start,inputIndexEnd:t.end},e),{})),null!=n.attrs&&(r.attrParams=n.attrs.reduce((t,n)=>{let r=n.type,a;switch(n.type){case"string":void 0===(a=eu(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=eu(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"string[]":void 0===(a=eb(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=eb(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"number":void 0===(a=ep(e.attr,n.tfName,n.defaultValue||0))&&n.tfDeprecatedName&&(a=ep(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"number[]":void 0===(a=ey(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=ey(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"bool":void 0===(a=el(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=el(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"bool[]":void 0===(a=eN(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=eN(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"shape":void 0===(a=eg(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=eg(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"shape[]":void 0===(a=ek(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=ek(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"dtype":void 0===(a=ed(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=ed(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"dtype[]":void 0===(a=ef(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=ef(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"func":void 0===(a=eh(e.attr,n.tfName,n.defaultValue))&&n.tfDeprecatedName&&(a=eh(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"tensor":case"tensors":break;default:throw Error(`Unsupported param type: ${n.type} for op: ${e.op}`)}return t[n.name]={value:a,type:r},t},{})),r}mapFunction(e){let t=e.nodeDef,n=[],r={};null!=t&&(r=t.reduce((e,t)=>(e[t.name]=this.mapNode(t),"Const"===t.op&&n.push(e[t.name]),e),{}));let a=[],s=[];e.signature.inputArg.forEach(e=>{let[t,]=O(e.name),n={name:t,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:ec(e.type),type:"dtype"}},children:[]};n.signatureKey=e.name,a.push(n),r[t]=n});let i=Object.keys(r);i.forEach(e=>{let t=r[e];t.inputNames.forEach((e,n)=>{let[a,,s]=O(e),i=r[a];if(null!=i.outputs){let o=i.outputs.indexOf(s);if(-1!==o){let u=`${a}:${o}`;t.inputNames[n]=u}}t.inputs.push(i),i.children.push(t)})});let o=e.ret;e.signature.outputArg.forEach(e=>{let[t,n]=O(o[e.name]),a=r[t];null!=a&&(a.defaultOutput=n,s.push(a))});let u=this.mapArgsToSignature(e);return{nodes:r,inputs:a,outputs:s,weights:n,placeholders:[],signature:u}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((e,t)=>(e[t.name]=this.mapArgToTensorInfo(t),e),{}),outputs:e.signature.outputArg.reduce((t,n)=>(t[n.name]=this.mapArgToTensorInfo(n,e.ret),t),{})}}mapArgToTensorInfo(e,t){let n=e.name;return null!=t&&(n=t[n]),{name:n,dtype:e.type}}}function eo(e,t){let n=Array.isArray(e)?String.fromCharCode.apply(null,e):function(e){let t=(0,E.env)().global;if(void 0!==t.atob)return t.atob(e);if(void 0!==es)return new es(e,"base64").toString();throw Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}(e);return t?n:n.toLowerCase()}function eu(e,t,n,r=!1){let a=e[t];return null!=a?eo(a.s,r):n}function el(e,t,n){let r=e[t];return r?r.b:n}function ep(e,t,n){let r=e[t]||{},a=null!=r.i?r.i:null!=r.f?r.f:n;return"number"==typeof a?a:parseInt(a,10)}function ec(e){switch("string"==typeof e&&(e=i[e]),e){case i.DT_FLOAT:case i.DT_HALF:return"float32";case i.DT_INT32:case i.DT_INT64:case i.DT_INT8:case i.DT_UINT8:return"int32";case i.DT_BOOL:return"bool";case i.DT_DOUBLE:return"float32";case i.DT_STRING:return"string";default:return null}}function eh(e,t,n){let r=e[t];return r&&r.func?r.func.name:n}function ed(e,t,n){let r=e[t];return r&&r.type?ec(r.type):n}function ef(e,t,n){let r=e[t];return r&&r.list&&r.list.type?r.list.type.map(e=>ec(e)):n}function em(e){return e.unknownRank?void 0:null!=e.dim?e.dim.map(e=>"number"==typeof e.size?e.size:parseInt(e.size,10)):[]}function eg(e,t,n){let r=e[t];return r&&r.shape?em(r.shape):n}function ey(e,t,n){let r=e[t];return r?((r.list.f&&r.list.f.length?r.list.f:r.list.i)||[]).map(e=>"number"==typeof e?e:parseInt(e,10)):n}function eb(e,t,n,r=!1){let a=e[t];return a&&a.list&&a.list.s?a.list.s.map(e=>eo(e,r)):n}function ek(e,t,n){let r=e[t];return r&&r.list&&r.list.shape?r.list.shape.map(e=>em(e)):n}function eN(e,t,n){let r=e[t];return r&&r.list&&r.list.b?r.list.b:n}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class ev{constructor(e,t,n){this.node=e,this.tensorMap=t,this.context=n,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(e=>this.getInput(e)),null!=e.rawAttrs&&(this.attrs=Object.keys(e.rawAttrs).reduce((e,t)=>(e[t]=this.getAttr(t),e),{}))}getInput(e){return B(e,this.tensorMap,this.context)}getAttr(e,t){let n=this.node.rawAttrs[e];if(null!=n.tensor)return B(e,this.tensorMap,this.context);if(null!=n.i||null!=n.f)return ep(this.node.rawAttrs,e,t);if(null!=n.s)return eu(this.node.rawAttrs,e,t);if(null!=n.b)return el(this.node.rawAttrs,e,t);if(null!=n.shape)return eg(this.node.rawAttrs,e,t);if(null!=n.type)return ed(this.node.rawAttrs,e,t);if(null!=n.list){if(null!=n.list.i||null!=n.list.f)return ey(this.node.rawAttrs,e,t);if(null!=n.list.s)return eb(this.node.rawAttrs,e,t);if(null!=n.list.shape)return ek(this.node.rawAttrs,e,t);if(null!=n.list.b)return eN(this.node.rawAttrs,e,t);if(null!=n.list.type)return ef(this.node.rawAttrs,e,t)}return t}}var ex=n(2071);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let ew=(e,t,n,r=_)=>{switch(e.op){case"BiasAdd":case"AddV2":case"Add":return[r.add(F("a",e,t,n),F("b",e,t,n))];case"AddN":return[r.addN(F("tensors",e,t,n))];case"FloorMod":case"Mod":return[r.mod(F("a",e,t,n),F("b",e,t,n))];case"Mul":return[r.mul(F("a",e,t,n),F("b",e,t,n))];case"RealDiv":case"Div":return[r.div(F("a",e,t,n),F("b",e,t,n))];case"DivNoNan":return[r.divNoNan(F("a",e,t,n),F("b",e,t,n))];case"FloorDiv":return[r.floorDiv(F("a",e,t,n),F("b",e,t,n))];case"Sub":return[r.sub(F("a",e,t,n),F("b",e,t,n))];case"Minimum":return[r.minimum(F("a",e,t,n),F("b",e,t,n))];case"Maximum":return[r.maximum(F("a",e,t,n),F("b",e,t,n))];case"Pow":return[r.pow(F("a",e,t,n),F("b",e,t,n))];case"SquaredDifference":return[r.squaredDifference(F("a",e,t,n),F("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},eT=(e,t,n,r=_)=>{switch(e.op){case"Abs":case"ComplexAbs":return[r.abs(F("x",e,t,n))];case"Acos":return[r.acos(F("x",e,t,n))];case"Acosh":return[r.acosh(F("x",e,t,n))];case"Asin":return[r.asin(F("x",e,t,n))];case"Asinh":return[r.asinh(F("x",e,t,n))];case"Atan":return[r.atan(F("x",e,t,n))];case"Atan2":return[r.atan2(F("x",e,t,n),F("y",e,t,n))];case"Atanh":return[r.atanh(F("x",e,t,n))];case"Ceil":return[r.ceil(F("x",e,t,n))];case"Complex":return[r.complex(F("real",e,t,n),F("imag",e,t,n))];case"Cos":return[r.cos(F("x",e,t,n))];case"Cosh":return[r.cosh(F("x",e,t,n))];case"Elu":return[r.elu(F("x",e,t,n))];case"Erf":return[r.erf(F("x",e,t,n))];case"Exp":return[r.exp(F("x",e,t,n))];case"Expm1":return[r.expm1(F("x",e,t,n))];case"Floor":return[r.floor(F("x",e,t,n))];case"Log":return[r.log(F("x",e,t,n))];case"Log1p":return[r.log1p(F("x",e,t,n))];case"Imag":return[r.imag(F("x",e,t,n))];case"Neg":return[r.neg(F("x",e,t,n))];case"Reciprocal":return[r.reciprocal(F("x",e,t,n))];case"Real":return[r.real(F("x",e,t,n))];case"Relu":return[r.relu(F("x",e,t,n))];case"Round":return[r.round(F("x",e,t,n))];case"Selu":return[r.selu(F("x",e,t,n))];case"Sigmoid":return[r.sigmoid(F("x",e,t,n))];case"Sin":return[r.sin(F("x",e,t,n))];case"Sign":return[r.sign(F("x",e,t,n))];case"Sinh":return[r.sinh(F("x",e,t,n))];case"Softplus":return[r.softplus(F("x",e,t,n))];case"Sqrt":return[r.sqrt(F("x",e,t,n))];case"Square":return[r.square(F("x",e,t,n))];case"Tanh":return[r.tanh(F("x",e,t,n))];case"Tan":return[r.tan(F("x",e,t,n))];case"ClipByValue":return[r.clipByValue(F("x",e,t,n),F("clipValueMin",e,t,n),F("clipValueMax",e,t,n))];case"Relu6":return[r.relu6(F("x",e,t,n))];case"Rsqrt":return[r.rsqrt(B(e.inputNames[0],t,n))];case"Prod":return[r.prod(F("x",e,t,n),F("axes",e,t,n))];case"LeakyRelu":return[r.leakyRelu(F("x",e,t,n),F("alpha",e,t,n))];case"Prelu":return[r.prelu(F("x",e,t,n),F("alpha",e,t,n))];case"IsNan":return[r.isNaN(B(e.inputNames[0],t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function eS(e,t,n=""){if("number"!=typeof e&&"number"!=typeof t){E.util.assert(e.length===t.length,()=>n+` Shapes ${e} and ${t} must match`);for(let r=0;r<e.length;r++){let a=e[r],s=t[r];E.util.assert(a<0||s<0||a===s,()=>n+` Shapes ${e} and ${t} must match`)}}}function eI(e){return!("number"==typeof e||e.some(e=>e<0))}function e_(e,t,n){let r=eE(e,n),a=!eI(r);if(a&&0===t.length)throw Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${r}`);if(a&&t.forEach(e=>{r=eE(e.shape,r)}),!eI(r))throw Error(`Non-fully-defined elementShape: ${r}`);return r}function eE(e,t){if("number"==typeof e)return t;if("number"==typeof t)return e;if(e.length!==t.length)throw Error(`Incompatible ranks during merge: ${e} vs. ${t}`);let n=[];for(let r=0;r<e.length;++r){let a=e[r],s=t[r];if(a>=0&&s>=0&&a!==s)throw Error(`Incompatible shape during merge: ${e} vs. ${t}`);n[r]=a>=0?a:s}return n}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class eA{constructor(e,t,n,r,a,s,i){this.name=e,this.dtype=t,this.maxSize=n,this.elementShape=r,this.identicalElementShapes=a,this.dynamicSize=s,this.clearAfterRead=i,this.tensors=[],this.closed_=!1,this.idTensor=(0,E.scalar)(0),(0,E.keep)(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{null!=e&&e.has(t.tensor.id)||t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);let t=this.tensors[e];if(t.cleared)throw Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(e=>this.read(e))}write(e,t){if(this.closed_)throw Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);let n=this.tensors[e]||{};if(t.dtype!==this.dtype)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(0===this.size()&&(null==this.elementShape||0===this.elementShape.length)&&(this.elementShape=t.shape),eS(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),n.read)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(n.written)throw Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);n.tensor=t,(0,E.keep)(t),n.written=!0,this.tensors[e]=n}writeMany(e,t){if(e.length!==t.length)throw Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((e,n)=>this.write(e,t[n]))}gather(e,t){if(t&&t!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let n=0;n<this.size();n++)e.push(n)}if(0===e.length)return(0,E.tensor)([],[0].concat(this.elementShape));let r=this.readMany(e);return eS(this.elementShape,r[0].shape,"TensorArray shape mismatch: "),(0,E.stack)(r,0)}concat(e){if(e&&e!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(0===this.size())return(0,E.tensor)([],[0].concat(this.elementShape));let t=[];for(let n=0;n<this.size();n++)t.push(n);let r=this.readMany(t);return eS(this.elementShape,r[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${r[0].shape})`),(0,E.concat)(r,0)}scatter(e,t){if(t.dtype!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);let n=Math.max(...e);if(!this.dynamicSize&&n>=this.maxSize)throw Error(`Max index must be < array size (${n}  vs. ${this.maxSize})`);this.writeMany(e,(0,E.unstack)(t,0))}split(e,t){if(t.dtype!==this.dtype)throw Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let n=0,r=e.map(e=>n+=e);if(n!==t.shape[0])throw Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${n}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);let a=0===n?0:t.size/n,s=[];(0,E.tidy)(()=>{t=(0,E.reshape)(t,[1,n,a]);for(let i=0;i<e.length;++i){let o=0===i?0:r[i-1],u=[0,o,0],l=[1,e[i],a];s[i]=(0,E.reshape)((0,E.slice)(t,u,l),this.elementShape)}return s});let i=[];for(let o=0;o<e.length;o++)i[o]=o;this.writeMany(i,s)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class eM{constructor(e,t,n,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=n,null!=e&&e.forEach(e=>{if(n!==e.dtype)throw Error(`Invalid data types; op elements ${n}, but list elements ${e.dtype}`);eS(t,e.shape,"TensorList shape mismatch: "),(0,E.keep)(e)}),this.idTensor=(0,E.scalar)(0),this.maxNumElements=r,(0,E.keep)(this.idTensor)}get id(){return this.idTensor.id}copy(){return new eM([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{null!=e&&e.has(t.id)||t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,n=-1){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(-1!==n&&this.tensors.length!==n)throw Error(`Operation expected a list with ${n} elements but got a list with ${this.tensors.length} elements.`);eS(e,this.elementShape,"TensorList shape mismatch: ");let r=e_(this.elementShape,this.tensors,e);return(0,E.tidy)(()=>{let e=this.tensors.map(e=>(0,E.reshape)(e,r));return(0,E.stack)(e,0)})}popBack(e,t){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(0===this.size())throw Error("Trying to pop from an empty list.");let n=e_(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,eS(r.shape,e,"TensorList shape mismatch: "),(0,E.reshape)(r,n)}pushBack(e){if(e.dtype!==this.elementDtype)throw Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(eS(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw Error("Trying to push element into a full list.");(0,E.keep)(e),this.tensors.push(e)}resize(e){if(e<0)throw Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(-1!==this.maxNumElements&&e>this.maxNumElements)throw Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);let t=new eM([],this.elementShape,this.elementDtype,this.maxNumElements);t.tensors.length=e;for(let n=0;n<Math.min(this.tensors.length,e);++n)t.tensors[n]=this.tensors[n];return t}getItem(e,t,n){if(n!==this.elementDtype)throw Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(null==this.tensors[e])throw Error(`element at index ${e} is null.`);eS(this.tensors[e].shape,t,"TensorList shape mismatch: ");let r=e_(this.elementShape,this.tensors,t);return(0,E.reshape)(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||-1!==this.maxNumElements&&e>=this.maxNumElements)throw Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);eS(this.elementShape,t.shape,"TensorList shape mismatch: "),(0,E.keep)(t),null!=this.tensors[e]&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,n){if(t!==this.elementDtype)throw Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);eS(this.elementShape,n,"TensorList shape mismatch: "),e=e.slice(0,this.size());let r=e_(this.elementShape,this.tensors,n);return 0===e.length?(0,E.tensor)([],[0].concat(r)):(0,E.tidy)(()=>{let t=e.map(e=>(0,E.reshape)(this.tensors[e],r));return(0,E.stack)(t,0)})}concat(e,t){if(e&&e!==this.elementDtype)throw Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);eS(this.elementShape,t,"TensorList shape mismatch: ");let n=e_(this.elementShape,this.tensors,t);return 0===this.size()?(0,E.tensor)([],[0].concat(n)):(0,E.tidy)(()=>{let e=this.tensors.map(e=>(0,E.reshape)(e,n));return(0,E.concat)(e,0)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let eD=async(e,t,n)=>{switch(e.op){case"If":case"StatelessIf":{let r=F("thenBranch",e,t,n),a=F("elseBranch",e,t,n),s=F("cond",e,t,n),i=F("args",e,t,n),o=await s.data();if(o[0])return n.functionMap[r].executeFunctionAsync(i,n.tensorArrayMap,n.tensorListMap);return n.functionMap[a].executeFunctionAsync(i,n.tensorArrayMap,n.tensorListMap)}case"While":case"StatelessWhile":{let u=F("body",e,t,n),l=F("cond",e,t,n),p=F("args",e,t,n),c=await n.functionMap[l].executeFunctionAsync(p,n.tensorArrayMap,n.tensorListMap),h=p.map(e=>e.id),d=await c[0].data();c.forEach(e=>{e.kept||-1!==h.indexOf(e.id)||e.dispose()});let f=p;for(;d[0];){let m=f;f=await n.functionMap[u].executeFunctionAsync(f,n.tensorArrayMap,n.tensorListMap);let g=f.map(e=>e.id);m.forEach(e=>{e.kept||-1!==h.indexOf(e.id)||-1!==g.indexOf(e.id)||e.dispose()});let y=await n.functionMap[l].executeFunctionAsync(f,n.tensorArrayMap,n.tensorListMap);d=await y[0].data(),y.forEach(e=>{e.kept||-1!==h.indexOf(e.id)||-1!==g.indexOf(e.id)||e.dispose()})}return f}case"LoopCond":{let b=F("pred",e,t,n);return[P(b)]}case"Switch":{let k=F("pred",e,t,n),N=F("data",e,t,n);return N.kept||(N=P(N)),(await k.data())[0]?[void 0,N]:[N,void 0]}case"Merge":{let v=e.inputNames.find(e=>void 0!==B(e,t,n));if(v){let x=B(v,t,n);return[P(x)]}return}case"Enter":{let w=F("frameName",e,t,n),T=F("tensor",e,t,n);return n.enterFrame(w),[P(T)]}case"Exit":{let S=F("tensor",e,t,n);return n.exitFrame(),[P(S)]}case"NextIteration":{let I=F("tensor",e,t,n);return n.nextIteration(),[P(I)]}case"TensorArrayV3":{let _=F("size",e,t,n),A=F("dtype",e,t,n),M=F("elementShape",e,t,n),D=F("dynamicSize",e,t,n),$=F("clearAfterRead",e,t,n),O=F("identicalElementShapes",e,t,n),R=F("name",e,t,n),C=new eA(R,A,_,M,O,D,$);return n.addTensorArray(C),[C.idTensor,(0,E.scalar)(1)]}case"TensorArrayWriteV3":{let V=F("tensorArrayId",e,t,n),L=F("index",e,t,n),z=F("tensor",e,t,n),W=n.getTensorArray(V.id);return W.write(L,z),[W.idTensor]}case"TensorArrayReadV3":{let U=F("tensorArrayId",e,t,n),G=F("index",e,t,n),q=n.getTensorArray(U.id);return[q.read(G)]}case"TensorArrayGatherV3":{let H=F("tensorArrayId",e,t,n),j=F("indices",e,t,n),K=F("dtype",e,t,n),X=n.getTensorArray(H.id);return[X.gather(j,K)]}case"TensorArrayScatterV3":{let Z=F("tensorArrayId",e,t,n),Q=F("indices",e,t,n),Y=F("tensor",e,t,n),J=n.getTensorArray(Z.id);return J.scatter(Q,Y),[J.idTensor]}case"TensorArrayConcatV3":{let ee=F("tensorArrayId",e,t,n),et=n.getTensorArray(ee.id),en=F("dtype",e,t,n);return[et.concat(en)]}case"TensorArraySplitV3":{let er=F("tensorArrayId",e,t,n),ea=F("tensor",e,t,n),es=F("lengths",e,t,n),ei=n.getTensorArray(er.id);return ei.split(es,ea),[ei.idTensor]}case"TensorArraySizeV3":{let eo=F("tensorArrayId",e,t,n),eu=n.getTensorArray(eo.id);return[(0,E.scalar)(eu.size(),"int32")]}case"TensorArrayCloseV3":{let el=F("tensorArrayId",e,t,n),ep=n.getTensorArray(el.id);return ep.clearAndClose(),[ep.idTensor]}case"TensorListSetItem":{let ec=F("tensorListId",e,t,n),eh=F("index",e,t,n),ed=F("tensor",e,t,n),ef=n.getTensorList(ec.id);return ef.setItem(eh,ed),[ef.idTensor]}case"TensorListGetItem":{let em=F("tensorListId",e,t,n),eg=F("index",e,t,n),ey=F("elementShape",e,t,n),eb=F("elementDType",e,t,n),ek=n.getTensorList(em.id);return[ek.getItem(eg,ey,eb)]}case"TensorListScatterV2":case"TensorListScatter":{let eN=F("indices",e,t,n),ev=F("tensor",e,t,n),ex=F("elementShape",e,t,n),ew=F("numElements",e,t,n),eT=function(e,t,n,r){if(t.length!==e.shape[0])throw Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${e.shape[0]}`);let a=Math.max(...t);if(null!=r&&-1!==r&&a>=r)throw Error(`Max index must be < array size (${a}  vs. ${r})`);let s=new eM([],n,e.dtype,r),i=(0,E.unstack)(e,0);return t.forEach((e,t)=>{s.setItem(e,i[t])}),s}(ev,eN,ex,ew);return n.addTensorList(eT),[eT.idTensor]}case"TensorListReserve":case"EmptyTensorList":{var eI,e_,eD;let e$=F("elementShape",e,t,n),eF=F("elementDType",e,t,n),eB;eB="TensorListReserve"===e.op?"numElements":"maxNumElements";let eO=F(eB,e,t,n),eR="TensorListReserve"===e.op?-1:eO,eC=new eM([],e$,eF,eR);return n.addTensorList(eC),[eC.idTensor]}case"TensorListGather":{let eV=F("tensorListId",e,t,n),eP=F("indices",e,t,n),eL=F("elementShape",e,t,n),ez=F("elementDType",e,t,n),eW=n.getTensorList(eV.id);return[eW.gather(eP,ez,eL)]}case"TensorListStack":{let eU=F("tensorListId",e,t,n),eG=F("elementShape",e,t,n),eq=F("elementDType",e,t,n),eH=F("numElements",e,t,n),ej=n.getTensorList(eU.id);return[ej.stack(eG,eq,eH)]}case"TensorListFromTensor":{let eK=F("tensor",e,t,n),eX=F("elementShape",e,t,n),eZ=F("elementDType",e,t,n),eQ=function(e,t,n){let r=e.dtype;if(e.shape.length<1)throw Error(`Tensor must be at least a vector, but saw shape: ${e.shape}`);if(e.dtype!==n)throw Error(`Invalid data types; op elements ${e.dtype}, but list elements ${n}`);let a=e.shape.slice(1);eS(a,t,"TensorList shape mismatch: ");let s=(0,E.unstack)(e);return new eM(s,t,r)}(eK,eX,eZ);return n.addTensorList(eQ),[eQ.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{let eY=F("tensorListId",e,t,n),eJ=n.getTensorList(eY.id),e0=F("dtype",e,t,n),e1=F("elementShape",e,t,n);return[eJ.concat(e0,e1)]}case"TensorListPushBack":{let e2=F("tensorListId",e,t,n),e3=F("tensor",e,t,n),e6=n.getTensorList(e2.id);return e6.pushBack(e3),[e6.idTensor]}case"TensorListPopBack":{let e4=F("tensorListId",e,t,n),e5=F("elementShape",e,t,n),e8=F("elementDType",e,t,n),e7=n.getTensorList(e4.id);return[e7.popBack(e5,e8)]}case"TensorListSplit":{let e9=F("tensor",e,t,n),te=F("elementShape",e,t,n),tt=F("lengths",e,t,n),tn=function(e,t,n){let r=0,a=t.map(e=>r+=e);if(r!==e.shape[0])throw Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${r}, and tensor's shape is: ${e.shape}`);let s=e.shape.slice(1),i=eE(s,n),o=0===r?0:e.size/r,u=(0,E.tidy)(()=>{let n=[];e=(0,E.reshape)(e,[1,r,o]);for(let s=0;s<t.length;++s){let u=0===s?0:a[s-1],l=[0,u,0],p=[1,t[s],o];n[s]=(0,E.reshape)((0,E.slice)(e,l,p),i)}return e.dispose(),n}),l=new eM([],n,e.dtype,t.length);for(let p=0;p<u.length;p++)l.setItem(p,u[p]);return l}(e9,tt,te);return n.addTensorList(tn),[tn.idTensor]}case"TensorListLength":{let tr=F("tensorListId",e,t,n),ta=n.getTensorList(tr.id);return[(0,E.scalar)(ta.size(),"int32")]}case"TensorListResize":{let ts=F("tensorListId",e,t,n),ti=F("size",e,t,n),to=n.getTensorList(ts.id),tu=to.resize(ti);return n.addTensorList(tu),[tu.idTensor]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function e$(e,t,n){let[r,a]=F("fusedOps",e,t,n),s="biasadd"===r,i="prelu"===a,o=F("numArgs",e,t,n);if(s){if(i&&2!==o)throw Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&s&&1!==o)throw Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if("fusedbatchnorm"===r)throw Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");let u=F("strides",e,t,n),l=V(e,t,n),p=F("dataFormat",e,t,n).toUpperCase(),c=F("dilations",e,t,n),[h,d]=F("args",e,t,n);s||(d=h,h=void 0);let f=F("leakyreluAlpha",e,t,n);return{stride:u,pad:l,dataFormat:p,dilations:c,biasArg:h,preluArg:d,activationFunc:a,leakyreluAlpha:f}}let eF=(e,t,n,r=_)=>{switch(e.op){case"Conv1D":{let a=F("stride",e,t,n),s=F("pad",e,t,n),i=F("dataFormat",e,t,n).toUpperCase(),o=F("dilation",e,t,n);return[r.conv1d(F("x",e,t,n),F("filter",e,t,n),a,s,i,o)]}case"Conv2D":{let u=F("strides",e,t,n),l=V(e,t,n),p=F("dataFormat",e,t,n).toUpperCase(),c=F("dilations",e,t,n);return[r.conv2d(F("x",e,t,n),F("filter",e,t,n),[u[1],u[2]],l,p,[c[1],c[2]])]}case"_FusedConv2D":{let{stride:h,pad:d,dataFormat:f,dilations:m,biasArg:g,preluArg:y,activationFunc:b,leakyreluAlpha:k}=e$(e,t,n);return[r.fused.conv2d({x:F("x",e,t,n),filter:F("filter",e,t,n),strides:[h[1],h[2]],pad:d,dataFormat:f,dilations:[m[1],m[2]],bias:g,activation:b,preluActivationWeights:y,leakyreluAlpha:k})]}case"FusedDepthwiseConv2dNative":{let{stride:N,pad:v,dataFormat:x,dilations:w,biasArg:T,preluArg:S,activationFunc:I,leakyreluAlpha:E}=e$(e,t,n);return[r.fused.depthwiseConv2d({x:F("x",e,t,n),filter:F("filter",e,t,n),strides:[N[1],N[2]],pad:v,dataFormat:x,dilations:[w[1],w[2]],bias:T,activation:I,preluActivationWeights:S,leakyreluAlpha:E})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{let A=F("outputShape",e,t,n),M=F("strides",e,t,n),D=V(e,t,n);return[r.conv2dTranspose(F("x",e,t,n),F("filter",e,t,n),A,[M[1],M[2]],D)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{let $=F("strides",e,t,n),B=V(e,t,n),O=F("dilations",e,t,n),R=F("dataFormat",e,t,n).toUpperCase();return[r.depthwiseConv2d(F("input",e,t,n),F("filter",e,t,n),[$[1],$[2]],B,R,[O[1],O[2]])]}case"Conv3D":{let C=F("strides",e,t,n),P=F("pad",e,t,n),L=F("dataFormat",e,t,n).toUpperCase(),z=F("dilations",e,t,n);return[r.conv3d(F("x",e,t,n),F("filter",e,t,n),[C[1],C[2],C[3]],P,L,[z[1],z[2],z[3]])]}case"AvgPool":{let W=F("strides",e,t,n),U=F("pad",e,t,n),G=F("kernelSize",e,t,n);return[r.avgPool(F("x",e,t,n),[G[1],G[2]],[W[1],W[2]],U)]}case"MaxPool":{let q=F("strides",e,t,n),H=F("pad",e,t,n),j=F("kernelSize",e,t,n);return[r.maxPool(F("x",e,t,n),[j[1],j[2]],[q[1],q[2]],H)]}case"MaxPoolWithArgmax":{let K=F("strides",e,t,n),X=F("pad",e,t,n),Z=F("kernelSize",e,t,n),Q=F("includeBatchInIndex",e,t,n),{result:Y,indexes:J}=r.maxPoolWithArgmax(F("x",e,t,n),[Z[1],Z[2]],[K[1],K[2]],X,Q);return[Y,J]}case"AvgPool3D":{let ee=F("strides",e,t,n),et=F("pad",e,t,n),en=F("kernelSize",e,t,n);return[r.avgPool3d(F("x",e,t,n),[en[1],en[2],en[3]],[ee[1],ee[2],ee[3]],et)]}case"MaxPool3D":{let er=F("strides",e,t,n),ea=F("pad",e,t,n),es=F("kernelSize",e,t,n);return[r.maxPool3d(F("x",e,t,n),[es[1],es[2],es[3]],[er[1],er[2],er[3]],ea)]}case"Dilation2D":{let ei=F("strides",e,t,n),eo=F("pad",e,t,n),eu=F("dilations",e,t,n),el=ei[1],ep=ei[2],ec=eu[1],eh=eu[2];return[r.dilation2d(F("x",e,t,n),F("filter",e,t,n),[el,ep],eo,[ec,eh],"NHWC")]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},eB=(e,t,n,r=_)=>{switch(e.op){case"Fill":{let a=F("shape",e,t,n),s=F("dtype",e,t,n),i=F("value",e,t,n);return[r.fill(a,i,s)]}case"LinSpace":{let o=F("start",e,t,n),u=F("stop",e,t,n),l=F("num",e,t,n);return[r.linspace(o,u,l)]}case"Multinomial":{let p=F("logits",e,t,n),c=F("numSamples",e,t,n),h=F("seed",e,t,n);return[r.multinomial(p,c,h)]}case"OneHot":{let d=F("indices",e,t,n),f=F("depth",e,t,n),m=F("onValue",e,t,n),g=F("offValue",e,t,n),y=F("dtype",e,t,n);return[r.oneHot(d,f,m,g,y)]}case"Ones":return[r.ones(F("shape",e,t,n),F("dtype",e,t,n))];case"OnesLike":return[r.onesLike(F("x",e,t,n))];case"RandomStandardNormal":return[r.randomStandardNormal(F("shape",e,t,n),F("dtype",e,t,n),F("seed",e,t,n))];case"RandomUniform":return[r.randomUniform(F("shape",e,t,n),F("minval",e,t,n),F("maxval",e,t,n),F("dtype",e,t,n))];case"Range":{let b=F("start",e,t,n),k=F("stop",e,t,n),N=F("step",e,t,n);return[r.range(b,k,N,F("dtype",e,t,n))]}case"TruncatedNormal":{let v=F("shape",e,t,n),x=F("mean",e,t,n),w=F("stdDev",e,t,n),T=F("seed",e,t,n);return[r.truncatedNormal(v,x,w,F("dtype",e,t,n),T)]}case"Zeros":return[r.zeros(F("shape",e,t,n),F("dtype",e,t,n))];case"ZerosLike":return[r.zerosLike(F("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function eO(e,t,n){let r=F("boxes",e,t,n),a=F("scores",e,t,n),s=F("maxOutputSize",e,t,n),i=F("iouThreshold",e,t,n),o=F("scoreThreshold",e,t,n),u=F("softNmsSigma",e,t,n);return{boxes:r,scores:a,maxOutputSize:s,iouThreshold:i,scoreThreshold:o,softNmsSigma:u}}let eR=async(e,t,n,r,a=_)=>{switch(e.op){case"NonMaxSuppressionV5":{let{boxes:s,scores:i,maxOutputSize:o,iouThreshold:u,scoreThreshold:l,softNmsSigma:p}=eO(e,t,n),c=await a.image.nonMaxSuppressionWithScoreAsync(s,i,o,u,l,p);return[c.selectedIndices,c.selectedScores]}case"NonMaxSuppressionV4":{let{boxes:h,scores:d,maxOutputSize:f,iouThreshold:m,scoreThreshold:g}=eO(e,t,n),y=F("padToMaxOutputSize",e,t,n),b=await a.image.nonMaxSuppressionPaddedAsync(h,d,f,m,g,y);return[b.selectedIndices,b.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{let{boxes:k,scores:N,maxOutputSize:v,iouThreshold:x,scoreThreshold:w}=eO(e,t,n);return[await a.image.nonMaxSuppressionAsync(k,N,v,x,w)]}case"Where":{let T=a.cast(F("condition",e,t,n),"bool"),S=[await a.whereAsync(T)];return T.dispose(),S}case"ListDiff":return a.setdiff1dAsync(F("x",e,t,n),F("y",e,t,n));default:throw TypeError(`Node type ${e.op} is not implemented`)}},eC=(e,t,n,r=_)=>{switch(e.op){case"LowerBound":{let a=F("sortedSequence",e,t,n),s=F("values",e,t,n);return[r.lowerBound(a,s)]}case"TopKV2":{let i=F("x",e,t,n),o=F("k",e,t,n),u=F("sorted",e,t,n),l=r.topk(i,o,u);return[l.values,l.indices]}case"UpperBound":{let p=F("sortedSequence",e,t,n),c=F("values",e,t,n);return[r.upperBound(p,c)]}case"Unique":{let h=F("x",e,t,n),d=r.unique(h);return[d.values,d.indices]}case"UniqueV2":{let f=F("x",e,t,n),m=F("axis",e,t,n),g=r.unique(f,m);return[g.values,g.indices]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},eV=(e,t,n,r=_)=>{switch(e.op){case"Const":return t[e.name];case"PlaceholderWithDefault":let a=F("default",e,t,n);return[B(e.name,t,n)||a];case"Placeholder":return[B(e.name,t,n)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{let s=F("x",e,t,n);return[P(s)]}case"IdentityN":return F("x",e,t,n).map(e=>P(e));case"Snapshot":let i=F("x",e,t,n);return[P(i)];case"Shape":return[r.tensor1d(F("x",e,t,n).shape,"int32")];case"ShapeN":return F("x",e,t,n).map(e=>r.tensor1d(e.shape));case"Size":return[r.scalar(F("x",e,t,n).size,"int32")];case"Rank":return[r.scalar(F("x",e,t,n).rank,"int32")];case"NoOp":return[r.scalar(1)];case"Print":let o=F("x",e,t,n),u=F("data",e,t,n),l=F("message",e,t,n),p=F("summarize",e,t,n);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(l);for(let c=0;c<u.length;c++)console.log(Array.prototype.slice.call(u[c].dataSync()).slice(0,p));return[o];default:throw TypeError(`Node type ${e.op} is not implemented`)}};var eP=n(9494);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class eL{constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=(0,E.scalar)(0),this.tensorMap=new Map,(0,E.keep)(this.handle)}get id(){return this.handle.id}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return eP.i(this.size(),"int32")}async import(e,t){this.checkKeyAndValueTensor(e,t);let n=await e.data();return this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),(0,E.tidy)(()=>{let e=(0,E.unstack)(t),r=n.length,a=e.length;E.util.assert(r===a,()=>`The number of elements doesn't match, keys has ${r} elements, the values has ${a} elements.`);for(let s=0;s<r;s++){let i=n[s],o=e[s];(0,E.keep)(o),this.tensorMap.set(i,o)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);let n=await e.data();return(0,E.tidy)(()=>{let e=[];for(let r=0;r<n.length;r++){let a=n[r],s=this.findWithDefault(a,t);e.push(s)}return(0,E.stack)(e)})}findWithDefault(e,t){let n=this.tensorMap.get(e);return null!=n?n:t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let ez=async(e,t,n,r)=>{switch(e.op){case"HashTable":case"HashTableV2":{let a=F("keyDType",e,t,n),s=F("valueDType",e,t,n),i=new eL(a,s);return r.addHashTable(e.name,i),[i.handle]}case"LookupTableImport":case"LookupTableImportV2":{let o=F("tableHandle",e,t,n,r),u=F("keys",e,t,n),l=F("values",e,t,n),p=r.getHashTableById(o.id);return[await p.import(u,l)]}case"LookupTableFind":case"LookupTableFindV2":{let c=F("tableHandle",e,t,n,r),h=F("keys",e,t,n),d=F("defaultValue",e,t,n),f=r.getHashTableById(c.id);return[await f.find(h,d)]}case"LookupTableSize":case"LookupTableSizeV2":{let m=F("tableHandle",e,t,n,r),g=r.getHashTableById(m.id);return[g.tensorSize()]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},eW=(e,t,n,r=_)=>{switch(e.op){case"ResizeBilinear":{let a=F("images",e,t,n),s=F("size",e,t,n),i=F("alignCorners",e,t,n),o=F("halfPixelCenters",e,t,n);return[r.image.resizeBilinear(a,[s[0],s[1]],i,o)]}case"ResizeNearestNeighbor":{let u=F("images",e,t,n),l=F("size",e,t,n),p=F("alignCorners",e,t,n),c=F("halfPixelCenters",e,t,n);return[r.image.resizeNearestNeighbor(u,[l[0],l[1]],p,c)]}case"CropAndResize":{let h=F("image",e,t,n),d=F("boxes",e,t,n),f=F("boxInd",e,t,n),m=F("cropSize",e,t,n),g=F("method",e,t,n),y=F("extrapolationValue",e,t,n);return[r.image.cropAndResize(h,d,f,m,g,y)]}case"ImageProjectiveTransformV3":{let b=F("images",e,t,n),k=F("transforms",e,t,n),N=F("outputShape",e,t,n),v=F("fillValue",e,t,n),x=F("interpolation",e,t,n),w=F("fillMode",e,t,n);return[r.image.transform(b,k,x.toLowerCase(),w.toLowerCase(),v,N)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},eU=(e,t,n,r=_)=>{switch(e.op){case"Equal":return[r.equal(F("a",e,t,n),F("b",e,t,n))];case"NotEqual":return[r.notEqual(F("a",e,t,n),F("b",e,t,n))];case"Greater":return[r.greater(F("a",e,t,n),F("b",e,t,n))];case"GreaterEqual":return[r.greaterEqual(F("a",e,t,n),F("b",e,t,n))];case"Less":return[r.less(F("a",e,t,n),F("b",e,t,n))];case"LessEqual":return[r.lessEqual(F("a",e,t,n),F("b",e,t,n))];case"LogicalAnd":return[r.logicalAnd(F("a",e,t,n),F("b",e,t,n))];case"LogicalNot":return[r.logicalNot(F("a",e,t,n))];case"LogicalOr":return[r.logicalOr(F("a",e,t,n),F("b",e,t,n))];case"Select":case"SelectV2":return[r.where(F("condition",e,t,n),F("a",e,t,n),F("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},eG=(e,t,n,r=_)=>{switch(e.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[r.matMul(F("a",e,t,n),F("b",e,t,n),F("transposeA",e,t,n),F("transposeB",e,t,n))];case"Einsum":return[r.einsum(F("equation",e,t,n),...F("tensors",e,t,n))];case"Transpose":return[r.transpose(F("x",e,t,n),F("perm",e,t,n))];case"_FusedMatMul":let[a,s]=F("fusedOps",e,t,n),i="prelu"===s,o=F("numArgs",e,t,n),u=F("leakyreluAlpha",e,t,n);if("biasadd"===a){if(i&&2!==o)throw Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&1!==o)throw Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}let[l,p]=F("args",e,t,n);return[r.fused.matMul({a:F("a",e,t,n),b:F("b",e,t,n),transposeA:F("transposeA",e,t,n),transposeB:F("transposeB",e,t,n),bias:l,activation:s,preluActivationWeights:p,leakyreluAlpha:u})];default:throw TypeError(`Node type ${e.op} is not implemented`)}},eq=(e,t,n,r=_)=>{switch(e.op){case"EuclideanNorm":return[r.euclideanNorm(F("x",e,t,n),F("axis",e,t,n),F("keepDims",e,t,n))];case"FusedBatchNorm":case"FusedBatchNormV2":case"FusedBatchNormV3":return[r.batchNorm(F("x",e,t,n),F("mean",e,t,n),F("variance",e,t,n),F("offset",e,t,n),F("scale",e,t,n),F("epsilon",e,t,n))];case"LRN":return[r.localResponseNormalization(F("x",e,t,n),F("radius",e,t,n),F("bias",e,t,n),F("alpha",e,t,n),F("beta",e,t,n))];case"Softmax":return[r.softmax(F("x",e,t,n))];case"LogSoftmax":return[r.logSoftmax(F("x",e,t,n))];case"SparseToDense":return[r.sparseToDense(F("sparseIndices",e,t,n),F("outputShape",e,t,n),F("sparseValues",e,t,n),F("defaultValue",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},eH=(e,t,n,r=_)=>{switch(e.op){case"Max":{let a=F("axis",e,t,n),s=F("keepDims",e,t,n);return[r.max(F("x",e,t,n),a,s)]}case"Mean":{let i=F("axis",e,t,n),o=F("keepDims",e,t,n);return[r.mean(F("x",e,t,n),i,o)]}case"Min":{let u=F("axis",e,t,n),l=F("keepDims",e,t,n);return[r.min(F("x",e,t,n),u,l)]}case"Sum":{let p=F("axis",e,t,n),c=F("keepDims",e,t,n);return[r.sum(F("x",e,t,n),p,c)]}case"All":{let h=F("axis",e,t,n),d=F("keepDims",e,t,n);return[r.all(F("x",e,t,n),h,d)]}case"Any":{let f=F("axis",e,t,n),m=F("keepDims",e,t,n);return[r.any(F("x",e,t,n),f,m)]}case"ArgMax":{let g=F("axis",e,t,n);return[r.argMax(F("x",e,t,n),g)]}case"ArgMin":{let y=F("axis",e,t,n);return[r.argMin(F("x",e,t,n),y)]}case"Prod":{let b=F("axis",e,t,n),k=F("keepDims",e,t,n);return[r.prod(F("x",e,t,n),b,k)]}case"Cumprod":{let N=F("axis",e,t,n),v=F("exclusive",e,t,n),x=F("reverse",e,t,n);return[r.cumprod(F("x",e,t,n),N,v,x)]}case"Cumsum":{let w=F("axis",e,t,n),T=F("exclusive",e,t,n),S=F("reverse",e,t,n);return[r.cumsum(F("x",e,t,n),w,T,S)]}case"Bincount":let I=F("x",e,t,n),E=F("weights",e,t,n),A=F("size",e,t,n);return[r.bincount(I,E,A)];case"DenseBincount":{let M=F("x",e,t,n),D=F("weights",e,t,n),$=F("size",e,t,n),B=F("binaryOutput",e,t,n);return[r.denseBincount(M,D,$,B)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},ej=(e,t,n,r=_)=>{switch(e.op){case"ConcatV2":case"Concat":{let a=F("n",e,t,n),s=F("axis",e,t,n),i=F("tensors",e,t,n);return i=i.slice(0,a),[r.concat(i,s)]}case"Gather":{let o=F("x",e,t,n),u=F("indices",e,t,n);return[r.gather(o,r.cast(u,"int32"),0)]}case"GatherV2":{let l=F("axis",e,t,n),p=F("batchDims",e,t,n),c=F("x",e,t,n),h=F("indices",e,t,n);return[r.gather(c,r.cast(h,"int32"),l,p)]}case"Reverse":{let d=F("dims",e,t,n),f=[];for(let m=0;m<d.length;m++)d[m]&&f.push(m);let g=F("x",e,t,n);return[r.reverse(g,f)]}case"ReverseV2":{let y=F("axis",e,t,n),b=F("x",e,t,n);return[r.reverse(b,y)]}case"Slice":{let k=F("begin",e,t,n),N=F("size",e,t,n);return[r.slice(F("x",e,t,n),k,N)]}case"StridedSlice":{let v=F("begin",e,t,n),x=F("end",e,t,n),w=F("strides",e,t,n),T=F("beginMask",e,t,n),S=F("endMask",e,t,n),I=F("ellipsisMask",e,t,n),A=F("newAxisMask",e,t,n),M=F("shrinkAxisMask",e,t,n),D=F("x",e,t,n);return[r.stridedSlice(D,v,x,w,T,S,I,A,M)]}case"Pack":return(0,E.tidy)(()=>{let a=F("axis",e,t,n),s=F("tensors",e,t,n),i=s[0].shape,o=r.squeeze(s[0]).shape,u=s.map(e=>{let t=E.util.arraysEqual(e.shape,i);if(!t&&!E.util.arraysEqual(r.squeeze(e).shape,o))throw Error("the input tensors shape does not match");return t?e:r.reshape(e,i)});return[r.stack(u,a)]});case"Unpack":{let $=F("axis",e,t,n),B=F("tensor",e,t,n);return r.unstack(B,$)}case"Tile":{let O=F("reps",e,t,n);return[r.tile(F("x",e,t,n),O)]}case"Split":case"SplitV":{let R=F("axis",e,t,n),C=F("numOrSizeSplits",e,t,n),V=F("x",e,t,n);return r.split(V,C,R)}case"ScatterNd":{let P=F("indices",e,t,n),L=F("values",e,t,n),z=F("shape",e,t,n);return[r.scatterND(P,L,z)]}case"GatherNd":{let W=F("x",e,t,n),U=F("indices",e,t,n);return[r.gatherND(W,U)]}case"SparseToDense":{let G=F("sparseIndices",e,t,n),q=F("outputShape",e,t,n),H=F("sparseValues",e,t,n),j=F("defaultValue",e,t,n);return[r.sparseToDense(G,H,q,H.dtype===j.dtype?j:r.cast(j,H.dtype))]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},eK=(e,t,n,r=_)=>{switch(e.op){case"SparseFillEmptyRows":{let{outputIndices:a,outputValues:s,emptyRowIndicator:i,reverseIndexMap:o}=r.sparse.sparseFillEmptyRows(F("indices",e,t,n),F("values",e,t,n),F("denseShape",e,t,n),F("defaultValue",e,t,n));return[a,s,i,o]}case"SparseReshape":{let{outputIndices:u,outputShape:l}=r.sparse.sparseReshape(F("inputIndices",e,t,n),F("inputShape",e,t,n),F("newShape",e,t,n));return[u,l]}case"SparseSegmentMean":{let p=r.sparse.sparseSegmentMean(F("data",e,t,n),F("indices",e,t,n),F("segmentIds",e,t,n));return[p]}case"SparseSegmentSum":{let c=r.sparse.sparseSegmentSum(F("data",e,t,n),F("indices",e,t,n),F("segmentIds",e,t,n));return[c]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},eX=(e,t,n,r=_)=>{switch(e.op){case"FFT":return[r.fft(F("x",e,t,n))];case"IFFT":return[r.ifft(F("x",e,t,n))];case"RFFT":return[r.rfft(F("x",e,t,n))];case"IRFFT":return[r.irfft(F("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}},eZ=(e,t,n,r=_)=>{switch(e.op){case"StringNGrams":{let{nGrams:a,nGramsSplits:s}=r.string.stringNGrams(F("data",e,t,n),F("dataSplits",e,t,n),F("separator",e,t,n),F("nGramWidths",e,t,n),F("leftPad",e,t,n),F("rightPad",e,t,n),F("padWidth",e,t,n),F("preserveShortSequences",e,t,n));return[a,s]}case"StringSplit":{let{indices:i,values:o,shape:u}=r.string.stringSplit(F("input",e,t,n),F("delimiter",e,t,n),F("skipEmpty",e,t,n));return[i,o,u]}case"StringToHashBucketFast":{let l=r.string.stringToHashBucketFast(F("input",e,t,n),F("numBuckets",e,t,n));return[l]}default:throw TypeError(`Node type ${e.op} is not implemented`)}},eQ=(e,t,n,r=_)=>{switch(e.op){case"Cast":return[r.cast(F("x",e,t,n),F("dtype",e,t,n))];case"ExpandDims":{let a=F("axis",e,t,n);return[r.expandDims(F("x",e,t,n),a)]}case"Squeeze":{let s=F("axis",e,t,n);return[r.squeeze(F("x",e,t,n),s)]}case"Reshape":return[r.reshape(F("x",e,t,n),F("shape",e,t,n))];case"MirrorPad":return[r.mirrorPad(F("x",e,t,n),F("padding",e,t,n),F("mode",e,t,n))];case"PadV2":case"Pad":return[r.pad(F("x",e,t,n),F("padding",e,t,n),F("constantValue",e,t,n))];case"SpaceToBatchND":{let i=F("blockShape",e,t,n),o=F("paddings",e,t,n);return[r.spaceToBatchND(F("x",e,t,n),i,o)]}case"BatchToSpaceND":{let u=F("blockShape",e,t,n),l=F("crops",e,t,n);return[r.batchToSpaceND(F("x",e,t,n),u,l)]}case"DepthToSpace":{let p=F("blockSize",e,t,n),c=F("dataFormat",e,t,n).toUpperCase();return[r.depthToSpace(F("x",e,t,n),p,c)]}case"BroadcastTo":return[r.broadcastTo(F("x",e,t,n),F("shape",e,t,n))];case"BroadcastArgs":return[r.broadcastArgs(F("s0",e,t,n),F("s1",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function eY(e,t,n,r,a=E.tidy){let s=((e,t,n)=>{switch(e.category){case"arithmetic":return a(()=>ew(e,t,n));case"basic_math":return a(()=>eT(e,t,n));case"control":return eD(e,t,n);case"convolution":return a(()=>eF(e,t,n));case"creation":return a(()=>eB(e,t,n));case"dynamic":return eR(e,t,n);case"evaluation":return a(()=>eC(e,t,n));case"image":return a(()=>eW(e,t,n));case"graph":return a(()=>eV(e,t,n));case"logical":return a(()=>eU(e,t,n));case"matrices":return a(()=>eG(e,t,n));case"normalization":return a(()=>eq(e,t,n));case"reduction":return a(()=>eH(e,t,n));case"slice_join":return a(()=>ej(e,t,n));case"sparse":return a(()=>eK(e,t,n));case"spectral":return a(()=>eX(e,t,n));case"string":return a(()=>eZ(e,t,n));case"transformation":return a(()=>eQ(e,t,n));case"hash_table":return ez(e,t,n,r);case"custom":var s;let i=M[s=e.op];if(i&&i.customExecutor)return i.customExecutor(new ev(e,t,n));throw TypeError(`Custom op ${e.op} is not registered.`);default:throw TypeError(`Unknown op '${e.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(e,t,n);return E.util.isPromise(s)?s.then(e=>[].concat(e)):[].concat(s)}class eJ{constructor(e={},t={},n={},r={}){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=n,this.functionMap=r,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){let e=[];for(let t=0;t<this.contexts.length-1;t++){let n=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(n))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(e=>0===e.id&&0===e.iterationId?"":`${e.frameName}-${e.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;let e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(let t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(let n in this.tensorListMap)this.tensorListMap[n].clearAndClose(e)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function e0(e,t,n,r){let a=new Set,s=[],i=null,o=null,u=new Set,l=Object.keys(e).map(e=>C(e)[0]),p=[];null!=r&&(p=r.map(e=>C(e.name)[0]));let c=[...t];for(;c.length>0;){let h=c.pop();if((e6(h)||e4(h)||e5(h))&&null==i&&(o=(i=h).children.map(e=>e.name).filter(e=>a.has(e))),a.add(h.name),null==n[h.name]&&-1===l.indexOf(h.name)&&-1===p.indexOf(h.name)){if(0===h.inputs.length){s.push(h.name);continue}h.inputs.forEach(e=>{!u.has(e.name)&&(u.add(e.name),c.push(e))})}}return{inputs:e,outputs:t,usedNodes:a,missingInputs:s,dynamicNode:i,syncInputs:o}}let e1=["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"],e2=["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"],e3=["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"];function e6(e){return e1.indexOf(e.op)>=0}function e4(e){return e2.indexOf(e.op)>=0}function e5(e){return e3.indexOf(e.op)>=0}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class e8{constructor(e,t){this.graph=e,this.parent=t,this.compiledMap=new Map,this._weightMap={},this.SEPERATOR=",",this._functions={},this._functionExecutorMap={},this.intermediateTensors={},this.keepTensorForDebug=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,null!=e.functions&&Object.keys(e.functions).forEach(t=>{this._functionExecutorMap[t]=new e8(e.functions[t],this)})}get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){let t=Object.keys(e).map(t=>e[t].map(e=>e.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{let t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}getCompilationKey(e,t){let n=e.map(e=>e.name).sort(),r=t.map(e=>e.name).sort();return n.join(this.SEPERATOR)+"--"+r.join(this.SEPERATOR)}compile(e,t){let n=e0(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:a,syncInputs:s}=n;if(null!=a)throw Error(`This execution contains the node '${a.name}', which has the dynamic op '${a.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${s}]`);if(r.length>0){let i=t.map(e=>e.name),o=Object.keys(e);throw Error(`Cannot compute the outputs [${i}] from the provided inputs [${o}]. Missing the following inputs: [${r}]`)}return function(e,t,n){let{usedNodes:r,inputs:a}=n,s=[],i=Object.keys(a).map(e=>C(e)[0]).map(t=>e.nodes[t]),o=e.initNodes;i.forEach(e=>{r.has(e.name)&&s.push(e)}),e.weights.forEach(e=>{r.has(e.name)&&s.push(e)}),null!=o&&o.forEach(e=>{r.has(e.name)&&s.push(e)});let u=new Set,l=[];for(;s.length>0;){let p=s.pop();u.add(p.name),t[p.name]||l.push(p),p.children.forEach(e=>{!u.has(e.name)&&r.has(e.name)&&e.inputs.every(e=>u.has(e.name))&&s.push(e)})}return l}(this.graph,this.weightMap,n)}execute(e,t){e=this.mapInputs(e);let n=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);let r=n.map(e=>this.graph.nodes[C(e)[0]]),a=t.map(e=>C(e)[0]),s=a.map(e=>this.graph.nodes[e]);this.resetIntermediateTensors(),0===s.length&&(s=this._outputs);let i=this.getCompilationKey(r,s),o=this.compiledMap.get(i);null==o&&(o=this.compile(e,s),this.compiledMap.set(i,o));let u={},l={};return(0,E.tidy)(()=>{let n=new eJ(this.weightMap,u,l,this.functionExecutorMap),r=Object.assign({},this.weightMap);Object.keys(e).forEach(t=>{let[n,a]=C(t),s=[];s[a]=e[t],r[n]=s});let s=this.getFrozenTensorIds(r),i={};for(let p=0;p<o.length;p++){let c=o[p];if(!r[c.name]){let h=eY(c,r,n,this._resourceManager);if(E.util.isPromise(h))throw Error(`The execution of the op '${c.op}' returned a promise. Please use model.executeAsync() instead.`);r[c.name]=h,this.checkTensorForDisposal(c.name,c,r,n,s,a,i)}}return null==this.parent&&n.dispose(s),t.map(e=>B(e,r,n))})}getFrozenTensorIds(e){let t=[].concat.apply([],Object.keys(e).map(t=>e[t]).map(e=>e.map(e=>e.id)));return new Set(t)}checkTensorForDisposal(e,t,n,r,a,s,i){"control"!==t.category&&-1===s.indexOf(e)&&(n[e].forEach(e=>{null!=e&&(i[e.id]=(i[e.id]||0)+t.children.length)}),t.inputs.forEach(e=>{if("control"!==e.category){var s,o,u;let l=n[R(s=e.name,r.currentContextId)];null!=l&&l.forEach(e=>{if(e&&!e.kept&&!a.has(e.id)){let n=i[e.id];if(1===n){if(this.keepTensorForDebug){let[s,o]=O(t.name,r);this.intermediateTensors[s]||(this.intermediateTensors[s]=[]),this.intermediateTensors[s][o]=e}else e.dispose();delete i[e.id]}else null!=n&&i[e.id]--}})}}))}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.intermediateTensors&&(Object.keys(this.intermediateTensors).forEach(e=>this.intermediateTensors[e].forEach(e=>e.dispose())),this.disposeTensorsMap())}disposeTensorsMap(){this.tensorsMap&&Object.keys(this.tensorsMap).forEach(e=>{let t=this.tensorsMap[e];t.forEach(e=>{!e||e.kept||e.isDisposed||this.keepIds.has(e.id)||e.dispose()})})}getIntermediateTensors(){return this.tensorsMap}resetIntermediateTensors(){for(let e in this.intermediateTensors)this.intermediateTensors[e].forEach(e=>e.dispose()),delete this.intermediateTensors[e]}async _executeAsync(e,t,n=!1,r={},a={}){n||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepTensorForDebug=(0,E.env)().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(s){console.warn(s.message)}this.resetIntermediateTensors();let i=new eJ(this.weightMap,r,a,this.functionExecutorMap);this.tensorsMap=await this.executeWithControlFlow(e,i,t,n);let o=t.map(e=>B(e,this.tensorsMap,i)),u=o.map(e=>e.id),l=Object.keys(e).map(t=>e[t].id);return this.keepIds=new Set([...u,...l,...this.weightIds]),this.keepTensorForDebug||this.disposeTensorsMap(),null==this.parent&&i.dispose(this.keepIds),o}async executeFunctionAsync(e,t,n){let r=e.reduce((e,t,n)=>(e[this.inputs[n].name]=t,e),{});return this._executeAsync(r,this.outputNodes,!0,t,n)}async executeWithControlFlow(e,t,n,r){let a=Object.keys(e),s=a.map(e=>this.graph.nodes[C(e)[0]]),i=n.map(e=>C(e)[0]),o=i.map(e=>this.graph.nodes[e]);0===o.length&&(o=this._outputs);let{usedNodes:u,missingInputs:l,dynamicNode:p,syncInputs:c}=e0(e,o,this.weightMap,this._initNodes),h=[...s,...this.graph.weights,...this._initNodes||[]].map(e=>({node:e,contexts:t.currentContext})),d=Object.assign({},this.weightMap);Object.keys(e).forEach(t=>{let[n,r]=C(t),a=[];a[r]=e[t],d[n]=a});let f={},m=this.getFrozenTensorIds(d),g={};for(;h.length>0;){let y=this.processStack(s,h,t,d,g,m,i,f,u);await Promise.all(y)}null!=p||r||console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");let b=o.filter(e=>!e6(e)&&!B(e.name,d,t)).map(e=>e.name);if(b.length>0){let k="";throw null!=p&&(k=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${c}]`),Error(`Cannot compute the outputs [${b}] from the provided inputs [${a}]. Consider providing the following inputs: [${l}]. ${k}`)}return d}processStack(e,t,n,r,a,s,i,o,u){let l=[];for(;t.length>0;){let p=t.pop();n.currentContext=p.contexts;let c="";if("Enter"===p.node.op&&F("isConstant",p.node,r,n)&&([c]=O(p.node.name,n)),null==r[p.node.name]){let h=eY(p.node,r,n,this._resourceManager);c||([c]=O(p.node.name,n));let d=n.currentContext;E.util.isPromise(h)?l.push(h.then(e=>(r[c]=e,n.currentContext=d,this.checkTensorForDisposal(c,p.node,r,n,s,i,o),this.processChildNodes(p.node,t,n,r,a,u),e))):(r[c]=h,this.checkTensorForDisposal(c,p.node,r,n,s,i,o),this.processChildNodes(p.node,t,n,r,a,u))}else this.processChildNodes(p.node,t,n,r,a,u)}return l}processChildNodes(e,t,n,r,a,s){e.children.forEach(e=>{let[i,]=O(e.name,n);!a[i]&&s.has(e.name)&&("Merge"===e.op?e.inputNames.some(e=>!!B(e,r,n))&&(a[i]=!0,t.push({contexts:n.currentContext,node:e})):e.inputNames.every(e=>!!B(e,r,n))&&(a[i]=!0,t.push({contexts:n.currentContext,node:e})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(e=>e.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{let n=e[t],[r,]=C(t),a=this.graph.nodes[r];if(a.attrParams.shape&&a.attrParams.shape.value){let s=a.attrParams.shape.value,i=s.length===n.shape.length&&n.shape.every((e,t)=>-1===s[t]||s[t]===e);E.util.assert(i,()=>`The shape of dict['${a.name}'] provided in model.execute(dict) must be [${s}], but was [${n.shape}]`)}a.attrParams.dtype&&a.attrParams.dtype.value&&E.util.assert(n.dtype===a.attrParams.dtype.value,()=>`The dtype of dict['${a.name}'] provided in model.execute(dict) must be ${a.attrParams.dtype.value}, but was ${n.dtype}`)})}mapInputs(e){let t={};for(let n in e)if(null!=this._signature&&null!=this._signature.inputs&&null!=this._signature.inputs[n]){let r=this._signature.inputs[n];t[r.name]=e[n]}else t[n]=e[n];return t}checkInputs(e){let t=Object.keys(e).filter(e=>{let[t]=C(e);return null==this.graph.nodes[t]});if(t.length>0)throw Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(e=>{if(null!=this._signature&&null!=this._signature.outputs&&null!=this._signature.outputs[e]){let t=this._signature.outputs[e];return t.name}return e},{})}checkOutputs(e){e.forEach(e=>{let[t]=C(e);if(!this.graph.nodes[t])throw Error(`The output '${e}' is not found in the graph`)})}}class e7{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(let e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(let t in this.hashTableNameToHandle)this.hashTableNameToHandle[t].dispose(),delete this.hashTableNameToHandle[t]}}class e9{constructor(e,t={},n=E.io){this.modelUrl=e,this.loadOptions=t,this.version="n/a",this.io=n,null==t&&(this.loadOptions={}),this.resourceManager=new e7}get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}findIOHandler(){let e=this.modelUrl;if(null!=e.load)this.handler=e;else if(null!=this.loadOptions.requestInit)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{let t=this.io.getLoadHandlers(e,this.loadOptions);if(0===t.length)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),null==this.handler.load)throw Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");let e=this.handler.load();return E.util.isPromise(e)?e.then(e=>this.loadSync(e)):this.loadSync(e)}loadSync(e){this.artifacts=e;let t=this.artifacts.modelTopology,n=this.artifacts.signature;if(null!=this.artifacts.userDefinedMetadata){let r=this.artifacts.userDefinedMetadata;null!=r.signature&&(n=r.signature),null!=r.structuredOutputKeys&&(this.structuredOutputKeys=r.structuredOutputKeys)}this.signature=n,this.version=`${t.versions.producer}.${t.versions.minConsumer}`;let a=this.io.decodeWeights(this.artifacts.weightData,this.artifacts.weightSpecs);if(this.executor=new e8(ei.Instance.transformGraph(t,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(a),this.executor.resourceManager=this.resourceManager,null!=e.modelInitializer&&null!=e.modelInitializer.node){let s=ei.Instance.transformGraph(e.modelInitializer);this.initializer=new e8(s),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializer.executeAsync({},[])}return!0}async save(e,t){if("string"==typeof e){let n=this.io.getSaveHandlers(e);if(0===n.length)throw Error(`Cannot find any save handlers for URL '${e}'`);if(n.length>1)throw Error(`Found more than one (${n.length}) save handlers for URL '${e}'`);e=n[0]}if(null==e.save)throw Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}predict(e,t){let n=this.execute(e,this.outputNodes);if(this.structuredOutputKeys){let r=n instanceof E.Tensor?[n]:n,a={};return r.forEach((e,t)=>a[this.structuredOutputKeys[t]]=e),a}return n}normalizeInputs(e){if(!(e instanceof E.Tensor)&&!Array.isArray(e))return e;if((e=Array.isArray(e)?e:[e]).length!==this.inputNodes.length)throw Error(`Input tensor count mismatch,the graph model has ${this.inputNodes.length} placeholders, while there are ${e.length} input tensors.`);return this.inputNodes.reduce((t,n,r)=>(t[n]=e[r],t),{})}normalizeOutputs(e){return Array.isArray(e=e||this.outputNodes)?e:[e]}execute(e,t){e=this.normalizeInputs(e),t=this.normalizeOutputs(t);let n=this.executor.execute(e,t);return n.length>1?n:n[0]}async executeAsync(e,t){e=this.normalizeInputs(e),t=this.normalizeOutputs(t);let n=await this.executor.executeAsync(e,t);return n.length>1?n:n[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,n)=>(t[n]=[e[n]],t),{})}dispose(){this.executor.dispose(),this.initializer&&this.initializer.dispose(),this.resourceManager.dispose()}}async function te(e,t={},n=E.io){var r;if(null==e)throw Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");null==t&&(t={}),t.fromTFHub&&"string"==typeof e&&(r=e,e=(r.endsWith("/")||(r+="/"),`${r}model.json?tfjs-format=file`));let a=new e9(e,t,n);return await a.load(),a}function tt(e){if(null==e)throw Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let t;if(e instanceof Array){let[n,r]=e;if(!n)throw Error("modelJSON must be the first element of the array");if(!r||!(r instanceof ArrayBuffer))throw Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in n))throw Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in n))throw Error("Model JSON is missing 'weightsManifest'");let a=E.io.getWeightSpecs(n.weightsManifest),s=E.io.getModelArtifactsForJSONSync(n,a,r);t=E.io.fromMemorySync(s)}else if("load"in e)t=e;else if("modelTopology"in e&&"weightSpecs"in e&&"weightData"in e)t=E.io.fromMemorySync(e);else throw Error("Unknown model format");let i=new e9(t);return i.load(),i}/** @license See the LICENSE file. */ let tn="3.21.0";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ },8713:function(e,t,n){"use strict";n.d(t,{JL:function(){return r},Zu:function(){return a}});class r{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class a{refCount(e){return s("refCount")}incRef(e){return s("incRef")}timerAvailable(){return!0}time(e){return s("time")}read(e){return s("read")}readSync(e){return s("readSync")}readToGPU(e,t){return s("readToGPU")}numDataIds(){return s("numDataIds")}disposeData(e,t){return s("disposeData")}write(e,t,n){return s("write")}move(e,t,n,r,a){return s("move")}createTensorFromTexture(e,t,n){return s("createTensorFromTexture")}memory(){return s("memory")}floatPrecision(){return s("floatPrecision")}epsilon(){return 32===this.floatPrecision()?1e-7:1e-4}dispose(){return s("dispose")}}function s(e){throw Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}},8329:function(e,t,n){"use strict";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function r(e,t,n){var r,s,i;let o=(r=e,s=t,i=n,function(e,t,n){let r=0,a=e.length,s=0,i=!1;for(;r<a;){s=r+(a-r>>>1);let o=n(t,e[s]);o>0?r=s+1:(a=s,i=!o)}return i?r:-r-1}(r,s,i||a));e.splice(o<0?-(o+1):o,0,t)}function a(e,t){return e>t?1:e<t?-1:0}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function s(e,t,n,r,a){return u(e,t,n,r,a,0)}function i(e,t,n,r,a,s){return u(e,t,n,r,a,0,!1,s,!0)}function o(e,t,n,r,a,s){return u(e,t,n,r,a,s,!0)}function u(e,t,n,a,s,i,o=!1,u=!1,h=!1){let d=[];for(let f=0;f<t.length;f++)t[f]>s&&d.push({score:t[f],boxIndex:f,suppressBeginIndex:0});d.sort(c);let m=i>0?-.5/i:0,g=[],y=[];for(;g.length<n&&d.length>0;){let b=d.pop(),{score:k,boxIndex:N,suppressBeginIndex:v}=b;if(k<s)break;let x=!1;for(let w=g.length-1;w>=v;--w){let T=l(e,N,g[w]);if(T>=a){x=!0;break}if(b.score=b.score*p(a,m,T),b.score<=s)break}b.suppressBeginIndex=g.length,!x&&(b.score===k?(g.push(N),y.push(b.score)):b.score>s&&r(d,b,c))}let S=g.length,I=n-S;u&&I>0&&(g.push(...Array(I).fill(0)),y.push(...Array(I).fill(0)));let _={selectedIndices:g};return o&&(_.selectedScores=y),h&&(_.validOutputs=S),_}function l(e,t,n){let r=e.subarray(4*t,4*t+4),a=e.subarray(4*n,4*n+4),s=Math.min(r[0],r[2]),i=Math.min(r[1],r[3]),o=Math.max(r[0],r[2]),u=Math.max(r[1],r[3]),l=Math.min(a[0],a[2]),p=Math.min(a[1],a[3]),c=Math.max(a[0],a[2]),h=Math.max(a[1],a[3]),d=(o-s)*(u-i),f=(c-l)*(h-p);if(d<=0||f<=0)return 0;let m=Math.max(s,l),g=Math.max(i,p),y=Math.min(o,c),b=Math.min(u,h),k=Math.max(y-m,0)*Math.max(b-g,0);return k/(d+f-k)}function p(e,t,n){let r=Math.exp(t*n*n);return n<=e?r:0}function c(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}n.d(t,{GP:function(){return s},qP:function(){return i},pA:function(){return o}})},8333:function(e,t,n){"use strict";n.d(t,{Z:function(){return a}});var r=n(2657);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function a(e,t){let n=[];for(let a=0;a<t.length;a++)t[a]&&n.push(a);let s=(0,r.f)(e,"int32"),i=(0,r.f)([n.length,e.length],"int32");for(let o=0;o<n.length;o++){let u=s.indexToLoc(n[o]),l=o*e.length;i.values.set(u,l)}return i.toTensor()}},196:function(e,t,n){"use strict";n.d(t,{BV:function(){return N},wv:function(){return k}});var r=n(8713),a=n(2885),s=n(5938),i=n(9121),o=n(6151),u=n(4706),l=n(3418),p=n(569);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class c{constructor(e,t){this.backendTimer=e,this.logger=t,null==t&&(this.logger=new d)}profileKernel(e,t,n){let r,s=()=>{r=n()},i,o=l.now();if(this.backendTimer.timerAvailable())i=this.backendTimer.time(s);else{for(let u of(s(),r))u.dataSync();i=Promise.resolve({kernelMs:l.now()-o})}if((0,a.OB)().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let p=0;p<r.length;p++){let c=r[p];c.data().then(t=>{h(t,c.dtype,e)})}let d={kernelName:e,outputs:r,inputs:t,timeMs:i.then(e=>e.kernelMs),extraInfo:i.then(e=>null!=e.getExtraProfileInfo?e.getExtraProfileInfo():"")};return d}logKernelProfile(e){let{kernelName:t,outputs:n,timeMs:r,inputs:a,extraInfo:s}=e;n.forEach(e=>{Promise.all([e.data(),r,s]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],a,n[2])})})}}function h(e,t,n){if("float32"!==t)return!1;for(let r=0;r<e.length;r++){let a=e[r];if(isNaN(a)||!isFinite(a))return console.warn(`Found ${a} in the result of '${n}'`),!0}return!1}class d{logKernelProfile(e,t,n,r,a,s){let i="number"==typeof r?p.oj(`${r}ms`,9):r.error,o=p.oj(e,25),u=t.rank,l=t.size,c=p.oj(t.shape.toString(),14),h="";for(let d in a){let f=a[d];if(null!=f){let m=f.shape||t.shape,g=m.length;h+=`${d}: ${g}D ${g>0?m:""} `}}console.log(`%c${o}	%c${i}	%c${u}D ${c}	%c${l}	%c${h}	%c${s}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}var f=n(974),m=n(747);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function g(e){return null!=e.kernelName}class y{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(let e in this.registeredVariables)this.registeredVariables[e].dispose()}}class b{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new y}async ready(){if(null!=this.pendingBackendInit)return this.pendingBackendInit.then(()=>{});if(null!=this.backendInstance)return;let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t],r=await this.initializeBackend(n).success;if(r){await this.setBackend(n);return}}throw Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(null!=this.pendingBackendInit)throw Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(null==this.backendInstance){let{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry)){if(!(e in this.registryFactory))return null;{let{asyncInit:t}=this.initializeBackend(e);if(t)return null}}return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(u.Z(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(null==this.registryFactory[e])throw Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,null==this.registry[e]){this.backendInstance=null;let{success:t,asyncInit:n}=this.initializeBackend(e),r=n?await t:t;if(!r)return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new c(this.backendInstance),!0}setupRegisteredKernels(){let e=(0,o.tr)(this.backendName);e.forEach(e=>{null!=e.setupFunc&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){let t=(0,o.tr)(e);t.forEach(t=>{null!=t.disposeFunc&&t.disposeFunc(this.registry[e])})}initializeBackend(e){let t=this.registryFactory[e];if(null==t)throw Error(`Cannot initialize backend ${e}, no registration found.`);try{let n=t.factory();if(!n||n instanceof r.Zu||"function"!=typeof n.then)return this.registry[e]=n,{success:!0,asyncInit:!1};{let a=++this.pendingBackendInitId,s=n.then(t=>!(a<this.pendingBackendInitId)&&(this.registry[e]=t,this.pendingBackendInit=null,!0)).catch(t=>!(a<this.pendingBackendInitId)&&(this.pendingBackendInit=null,u.Z(`Initialization of backend ${e} failed`),u.Z(t.stack||t.message),!1));return this.pendingBackendInit=s,{success:s,asyncInit:!0}}}catch(i){return u.Z(`Initialization of backend ${e} failed`),u.Z(i.stack||i.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw Error(`${e} backend not found in registry`);this.backendName===e&&null!=this.pendingBackendInit&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(0===Object.keys(this.registryFactory).length)throw Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t],{success:r,asyncInit:a}=this.initializeBackend(n);if(a||r)return{name:n,asyncInit:a}}throw Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,t){let n=this.state.tensorInfo.get(t),r=n.backend,a=this.readSync(t),s=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,a,n.shape,n.dtype,s),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(null==t){if("function"!=typeof e)throw Error("Please provide a function to tidy()");t=e}else{if("string"!=typeof e&&!(e instanceof String))throw Error("When calling with two arguments, the first argument to tidy() must be a string");if("function"!=typeof t)throw Error("When calling with two arguments, the 2nd argument to tidy() must be a function");n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>((r=t())instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(e,t,n){e();try{let r=n();return t(),r}catch(a){throw t(),a}}nextTensorId(){return b.nextTensorId++}nextVariableId(){return b.nextVariableId++}clone(e){let t=N.runKernel(i.iJz,{x:e}),n=e=>({x:()=>N.runKernel(i.RFZ,{x:e},{dtype:"float32"})});return this.addTapeNode(this.state.activeScope.name,{x:e},[t],n,[],{}),t}runKernel(e,t,n){null==this.backendName&&this.backend;let r=null!=(0,o.pI)(e,this.backendName);if(!r)throw Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,t,n){let r=this.backend.numDataIds(),a=0;n.forEach(e=>{a+="complex64"===e.dtype?3:1});let s=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=r-t-a-s;if(i>0)throw Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[],r=this.isTapeOn(),a=this.state.numBytes,s=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;null==this.backendName&&this.backend;let u,l=g(e)?e.kernelName:null!=this.state.activeScope?this.state.activeScope.name:"";if(g(e)){let{kernelName:c,inputs:h,attrs:d}=e;null==this.backendName&&this.backend;let f=(0,o.pI)(c,this.backendName);p.hu(null!=f,()=>`Cannot find registered kernel '${c}' for backend '${this.backendName}'`),i=()=>{let e=this.backend.numDataIds();u=f.kernelFunc({inputs:h,attrs:d,backend:this.backend});let t=Array.isArray(u)?u:[u];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,e,t);let a=t.map(e=>null!=e.rank?e:this.makeTensorFromTensorInfo(e));if(r){let s=this.getTensorsForGradient(c,h,a);n=this.saveTensorsForBackwardMode(s)}return a}}else{let{forwardFunc:m}=e,y=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};i=()=>{let e=this.backend.numDataIds();u=this.tidy(()=>m(this.backend,y));let t=Array.isArray(u)?u:[u];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,e,t),t}}let{inputs:b,attrs:k}=e,N=g(e)?null:e.backwardsFunc,v;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{this.ENV.getBool("DEBUG")||this.state.profiling?(v=this.profiler.profileKernel(l,b,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(v),t=v.outputs):t=i()}),r&&this.addTapeNode(l,b,t,N,n,k),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-a,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-s,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(b).map(e=>null!=b[e]?b[e].shape:null),outputShapes:t.map(e=>e.shape),kernelTimeMs:v.timeMs,extraInfo:v.extraInfo}),Array.isArray(u)?t:t[0]}saveTensorsForBackwardMode(e){let t=e.map(e=>this.keep(this.clone(e)));return t}getTensorsForGradient(e,t,n){let r=(0,o.uk)(e);if(null!=r){let a=r.inputsToSave||[],s=r.outputsToSave||[],i;r.saveAllInputs?(p.hu(Array.isArray(t),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(t).map(e=>t[e])):i=a.map(e=>t[e]);let u=n.filter((e,t)=>s[t]);return i.concat(u)}return[]}makeTensor(e,t,n,r){if(null==e)throw Error("Values passed to engine.makeTensor() are null");n=n||"float32",r=r||this.backend;let a=e;"string"===n&&p.HD(e[0])&&(a=e.map(e=>l.encodeString(e)));let s=r.write(a,t,n),i=new f.es(t,n,s,this.nextTensorId());if(this.trackTensor(i,r),"string"===n){let o=this.state.tensorInfo.get(s),u=(0,p.Ub)(a);this.state.numBytes+=u-o.bytes,o.bytes=u}return i}makeTensorFromDataId(e,t,n,r){n=n||"float32";let a={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(a,r)}makeTensorFromTensorInfo(e,t){let{dataId:n,shape:r,dtype:a}=e,s=new f.es(r,a,n,this.nextTensorId());return this.trackTensor(s,t),s}makeVariable(e,t=!0,n,r){n=n||this.nextVariableId().toString(),null!=r&&r!==e.dtype&&(e=e.cast(r));let a=new f._w(e,t,n,this.nextTensorId());if(null!=this.state.registeredVariables[a.name])throw Error(`Variable with name ${a.name} was already registered`);return this.state.registeredVariables[a.name]=a,this.incRef(a,this.backend),a}trackTensor(e,t){this.state.numTensors++,"string"===e.dtype&&this.state.numStringTensors++;let n=0;"complex64"!==e.dtype&&"string"!==e.dtype&&(n=e.size*p.bT(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof f._w||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;let t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,"string"===e.dtype&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),"complex64"!==e.dtype&&"string"!==e.dtype){let n=e.size*p.bT(e.dtype);this.state.numBytes-=n}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(let e in this.state.registeredVariables){let t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),null!=this.state.registeredVariables[e.name]&&delete this.state.registeredVariables[e.name]}memory(){let e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,null==e.reasons&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;let t=this.state.numBytes,n=this.state.numTensors;for(let r of(this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n,this.state.activeProfile.kernels))r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&0===this.state.kernelDepth}addTapeNode(e,t,n,r,a,s){let i={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:a},u=(0,o.uk)(e);null!=u&&(r=u.gradFunc),null!=r&&(i.gradient=e=>r((e=e.map((e,t)=>{if(null==e){let r=n[t],a=p.wT(r.size,r.dtype);return this.makeTensor(a,r.shape,r.dtype)}return e})).length>1?e:e[0],a,s)),this.state.activeTape.push(i)}keep(e){return e.kept=!0,e}startTape(){0===this.state.gradientDepth&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){let t={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){let t=(0,m.getTensorsInContainer)(e),n=new Set(t.map(e=>e.id));for(let r=0;r<this.state.activeScope.track.length;r++){let a=this.state.activeScope.track[r];a.kept||n.has(a.id)||a.dispose()}let s=this.state.scopeStack.pop();this.state.activeScope=0===this.state.scopeStack.length?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{e.kept||e.scopeId!==s.id||this.track(e)})}gradients(e,t,n,r=!1){if(p.hu(t.length>0,()=>"gradients() received an empty list of xs."),null!=n&&"float32"!==n.dtype)throw Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);let a=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));p.hu(a instanceof f.es,()=>"The result y returned by f() must be a tensor.");let s=/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r={},a={};for(let s=0;s<t.length;s++)r[t[s].id]=!0;for(let i=0;i<e.length;i++){let o=e[i],u=o.inputs;for(let l in u){let p=u[l],c=!1;for(let h=0;h<t.length;h++)if(r[p.id]){o.outputs.forEach(e=>r[e.id]=!0),c=!0,a[o.id]=!0;break}if(c)break}}let d={};d[n.id]=!0;let f={};for(let m=e.length-1;m>=0;m--){let g=e[m],y=g.inputs;for(let b=0;b<g.outputs.length;b++)if(d[g.outputs[b].id]){for(let k in y)d[y[k].id]=!0,f[g.id]=!0;break}}let N=[];for(let v=0;v<e.length;v++){let x=e[v];if(a[x.id]&&f[x.id]){let w={};for(let T in x.inputs){let S=x.inputs[T];r[S.id]&&(w[T]=S)}let I=Object.assign({},x);I.inputs=w,I.outputs=x.outputs,N.push(I)}}return N}(this.state.activeTape,t,a);if(!r&&0===s.length&&t.length>0)throw Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{let e={};e[a.id]=null==n?function(e){let t=(0,p.p8)((0,p.NA)(e),"float32");return N.makeTensor(t,e,"float32")}(a.shape):n,function(e,t,n,r){for(let a=t.length-1;a>=0;a--){let s=t[a],i=[];if(s.outputs.forEach(t=>{let n=e[t.id];null!=n?i.push(n):i.push(null)}),null==s.gradient)throw Error(`Cannot compute gradient: gradient function not found for ${s.kernelName}.`);let o=s.gradient(i);for(let u in s.inputs){if(!(u in o))throw Error(`Cannot backprop through input ${u}. Available gradients found: ${Object.keys(o)}.`);let l=n(()=>o[u]());if("float32"!==l.dtype)throw Error(`Error in gradient for op ${s.kernelName}. The gradient of input ${u} must have 'float32' dtype, but has '${l.dtype}'`);let c=s.inputs[u];if(!p.cO(l.shape,c.shape))throw Error(`Error in gradient for op ${s.kernelName}. The gradient of input '${u}' has shape '${l.shape}', which does not match the shape of the input '${c.shape}'`);if(null==e[c.id])e[c.id]=l;else{let h=e[c.id];e[c.id]=r(h,l),h.dispose()}}}}(e,s,e=>this.tidy(e),v);let r=t.map(t=>e[t.id]);return 0===this.state.gradientDepth&&(this.state.activeTape.forEach(e=>{for(let t of e.saved)t.dispose()}),this.state.activeTape=null),{value:a,grads:r}})}customGrad(e){return p.hu(p.mf(e),()=>"The f passed in customGrad(f) must be a function."),(...t)=>{p.hu(t.every(e=>e instanceof f.es),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let n,r={};t.forEach((e,t)=>{r[t]=e});let a=(r,a)=>(n=e(...[...t,a]),p.hu(n.value instanceof f.es,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),p.hu(p.mf(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),s=(e,r)=>{let a=n.gradFunc(e,r),s=Array.isArray(a)?a:[a];p.hu(s.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),p.hu(s.every(e=>e instanceof f.es),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");let i={};return s.forEach((e,t)=>{i[t]=()=>e}),i};return this.runKernelFunc({forwardFunc:a,backwardsFunc:s,inputs:r})}}readSync(e){let t=this.state.tensorInfo.get(e);return t.backend.readSync(e)}read(e){let t=this.state.tensorInfo.get(e);return t.backend.read(e)}readToGPU(e,t){let n=this.state.tensorInfo.get(e);return n.backend.readToGPU(e,t)}async time(e){let t=(0,l.now)(),n=await this.backend.time(e);return n.wallMs=(0,l.now)()-t,n}track(e){return null!=this.state.activeScope&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){for(let e in this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new y,this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}function k(){let e=(0,s.D)();if(null==e._tfengine){let t=new a.qA(e);e._tfengine=new b(t)}return(0,a.iG)(e._tfengine.ENV),(0,f.Vi)(()=>e._tfengine),e._tfengine}b.nextTensorId=0,b.nextVariableId=0;let N=k();function v(e,t){return N.runKernel(i.mm_,{a:e,b:t})}},2885:function(e,t,n){"use strict";n.d(t,{OB:function(){return o},Vi:function(){return u},iG:function(){return l},qA:function(){return s}});var r=n(569);/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let a="tfjsflags";class s{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=i,this.populateURLFlags()}setPlatform(e,t){null==this.platform||u.getBool("IS_TEST")||u.getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},null!=this.urlFlags[e]){let r=this.urlFlags[e];u.getBool("IS_TEST")||u.getBool("PROD")||console.warn(`Setting feature override from URL ${e}: ${r}.`),this.set(e,r)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];let t=this.evaluateFlag(e);if((0,r.tI)(t))throw Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(null==this.flagRegistry[e])throw Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,null!=this.flagRegistry[e].setHook&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(null==this.flagRegistry[e])throw Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(void 0===this.global||void 0===this.global.location||void 0===this.global.location.search)return;let e=this.getQueryParams(this.global.location.search);if(a in e){let t=e[a].split(",");t.forEach(e=>{let[t,n]=e.split(":");this.urlFlags[t]=function(e,t){if("true"===(t=t.toLowerCase())||"false"===t)return"true"===t;if(`${+t}`===t)return+t;throw Error(`Could not parse value flag value ${t} for flag ${e}.`)}(t,n)})}}}function i(e){let t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>{var r,a,s;return r=t,a=n[0],s=n[1],r[decodeURIComponent(a)]=decodeURIComponent(s||""),n.join("=")}),t}function o(){return u}let u=null;function l(e){u=e}},5938:function(e,t,n){"use strict";n.d(t,{D:function(){return s},R:function(){return i}});var r=n(3454);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let a;function s(){if(null==a){let e;if("undefined"!=typeof window)e=window;else if(void 0!==n.g)e=n.g;else if(void 0!==r)e=r;else if("undefined"!=typeof self)e=self;else throw Error("Could not find a global object");a=e}return a}function i(e,t){let n=function(){let e=s();return null==e._tfGlobals&&(e._tfGlobals=new Map),e._tfGlobals}();if(n.has(e))return n.get(e);{let r=t();return n.set(e,r),n.get(e)}}},4368:function(e,t,n){"use strict";n.d(t,{B9:function(){return g},CQ:function(){return k},Cd:function(){return N},Cn:function(){return y},G4:function(){return o},MX:function(){return p},N5:function(){return f},N8:function(){return c},N_:function(){return v},R:function(){return u},SR:function(){return h},VY:function(){return _},XV:function(){return b},cF:function(){return l},cj:function(){return x},jq:function(){return S},lu:function(){return m},sq:function(){return d},x3:function(){return w},y3:function(){return I},ze:function(){return T}});var r=n(196),a=n(2885),s=n(974),i=n(747);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function o(){(0,a.OB)().set("PROD",!0)}function u(){(0,a.OB)().set("DEBUG",!0)}function l(){(0,a.OB)().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function p(e){(0,a.OB)().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(e+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function c(){r.BV.disposeVariables()}function h(){return r.BV}function d(){return r.BV.memory()}function f(e){return r.BV.profile(e)}function m(e,t){return r.BV.tidy(e,t)}function g(e){let t=(0,i.getTensorsInContainer)(e);t.forEach(e=>e.dispose())}function y(e){return r.BV.keep(e)}function b(e){return r.BV.time(e)}function k(e){return r.BV.setBackend(e)}function N(){return r.BV.ready()}function v(){return r.BV.backendName}function x(e){r.BV.removeBackend(e)}function w(e){return r.BV.findBackend(e)}function T(e){return r.BV.findBackendFactory(e)}function S(e,t,n=1){return r.BV.registerBackend(e,t,n)}function I(){return r.BV.backend}function _(e,t){(0,a.OB)().setPlatform(e,t)}(0,s.FZ)(p)},633:function(e,t,n){"use strict";n.d(t,{UQ:function(){return o},cb:function(){return h},fN:function(){return p},h7:function(){return l},pn:function(){return c},ti:function(){return u}});var r=n(196),a=n(974),s=n(3740),i=n(569);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function o(e){return i.hu(i.mf(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{let a=(0,s._1)(t,"x","tf.grad","string_or_numeric"),o=null!=n?(0,s._1)(n,"dy","tf.grad"):null;return r.BV.tidy(()=>{let{value:t,grads:n}=r.BV.gradients(()=>e(a),[a],o);return null!=o&&i.k5(t.shape,o.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),d(n),n[0]})}}function u(e){return i.hu(i.mf(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{i.hu(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");let a=(0,s.sI)(t,"args","tf.grads","string_or_numeric"),o=null!=n?(0,s._1)(n,"dy","tf.grads"):null;return r.BV.tidy(()=>{let{value:t,grads:n}=r.BV.gradients(()=>e(...a),a,o);return null!=o&&i.k5(t.shape,o.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),d(n),n})}}function l(e){return i.hu(i.mf(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{i.hu(t instanceof a.es,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),i.hu(null==n||n instanceof a.es,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");let{grads:s,value:o}=r.BV.gradients(()=>e(t),[t],n);return d(s),{grad:s[0],value:o}}}function p(e){return i.hu(i.mf(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{i.hu(Array.isArray(t)&&t.every(e=>e instanceof a.es),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),i.hu(null==n||n instanceof a.es,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");let s=r.BV.gradients(()=>e(...t),t,n);return null!=n&&i.k5(s.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),d(s.grads),s}}function c(e,t){i.hu(i.mf(e),()=>"The f passed in variableGrads(f) must be a function"),i.hu(null==t||Array.isArray(t)&&t.every(e=>e instanceof a._w),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");let n=null!=t;if(!n)for(let s in t=[],r.BV.registeredVariables)t.push(r.BV.registeredVariables[s]);let o=n?t.filter(e=>!e.trainable):null,u=t.length;t=t.filter(e=>e.trainable),i.hu(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${u} variables is trainable.`);let{value:l,grads:p}=r.BV.gradients(e,t,null,!0);i.hu(p.some(e=>null!=e),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),i.hu(0===l.rank,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${l.rank} tensor`);let c={};return t.forEach((e,t)=>{null!=p[t]&&(c[e.name]=p[t])}),null!=o&&o.forEach(e=>c[e.name]=null),{value:l,grads:c}}function h(e){return r.BV.customGrad(e)}function d(e){let t=e.filter(e=>null==e).length;if(t>0)throw Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}},5793:function(e,t,n){"use strict";n.r(t),n.d(t,{Abs:function(){return eQ.SYM},Acos:function(){return eQ.VGw},Acosh:function(){return eQ.SpW},AdadeltaOptimizer:function(){return tW},AdagradOptimizer:function(){return tG},AdamOptimizer:function(){return tj},AdamaxOptimizer:function(){return tZ},Add:function(){return eQ.mm_},AddN:function(){return eQ.Xze},All:function(){return eQ.oT6},Any:function(){return eQ.IKK},ArgMax:function(){return eQ.sJF},ArgMin:function(){return eQ.aJk},Asin:function(){return eQ.M2y},Asinh:function(){return eQ.qw7},Atan:function(){return eQ.jMg},Atan2:function(){return eQ.QCc},Atanh:function(){return eQ.Oyi},AvgPool:function(){return eQ.JhU},AvgPool3D:function(){return eQ._k9},AvgPool3DGrad:function(){return eQ.IMb},AvgPoolGrad:function(){return eQ.ROF},BatchMatMul:function(){return eQ.XLW},BatchToSpaceND:function(){return eQ.zws},Bincount:function(){return eQ.zvY},BroadcastArgs:function(){return eQ.eEB},BroadcastTo:function(){return eQ.Ly9},Cast:function(){return eQ.RFZ},Ceil:function(){return eQ.gJX},ClipByValue:function(){return eQ.xnO},Complex:function(){return eQ.Zz9},ComplexAbs:function(){return eQ.yj2},Concat:function(){return eQ.Eh3},Conv2D:function(){return eQ.mhS},Conv2DBackpropFilter:function(){return eQ.wUP},Conv2DBackpropInput:function(){return eQ.wm},Conv3D:function(){return eQ.x12},Conv3DBackpropFilterV2:function(){return eQ.o2y},Conv3DBackpropInputV2:function(){return eQ.ik2},Cos:function(){return eQ.mc4},Cosh:function(){return eQ.TR1},CropAndResize:function(){return eQ.VcC},Cumprod:function(){return eQ.Byc},Cumsum:function(){return eQ.iHb},DataStorage:function(){return n4.JL},DenseBincount:function(){return eQ.QRR},DepthToSpace:function(){return eQ.T0n},DepthwiseConv2dNative:function(){return eQ.cie},DepthwiseConv2dNativeBackpropFilter:function(){return eQ.sL$},DepthwiseConv2dNativeBackpropInput:function(){return eQ.y7R},Diag:function(){return eQ.$w},Dilation2D:function(){return eQ.p4S},Dilation2DBackpropFilter:function(){return eQ.Vn9},Dilation2DBackpropInput:function(){return eQ.ekb},ENV:function(){return N.Vi},Einsum:function(){return eQ.$g6},Elu:function(){return eQ.SX0},EluGrad:function(){return eQ.HEU},Environment:function(){return N.qA},Equal:function(){return eQ.hdR},Erf:function(){return eQ.Omj},Exp:function(){return eQ.NEP},ExpandDims:function(){return eQ.YFo},Expm1:function(){return eQ.Y0y},FFT:function(){return eQ.vwp},Fill:function(){return eQ.deh},FlipLeftRight:function(){return eQ.Uyb},Floor:function(){return eQ.OR},FloorDiv:function(){return eQ.jeX},FromPixels:function(){return eQ.eBW},FusedBatchNorm:function(){return eQ.sHE},FusedConv2D:function(){return eQ._V0},FusedDepthwiseConv2D:function(){return eQ.luS},GatherNd:function(){return eQ.q1x},GatherV2:function(){return eQ.qi_},Greater:function(){return eQ.iZT},GreaterEqual:function(){return eQ.Acj},IFFT:function(){return eQ.Qg5},Identity:function(){return eQ.iJz},Imag:function(){return eQ.J_u},IsFinite:function(){return eQ.avt},IsInf:function(){return eQ.iWB},IsNan:function(){return eQ.r7n},KernelBackend:function(){return n4.Zu},LRN:function(){return eQ.eZ0},LRNGrad:function(){return eQ.Hhh},LeakyRelu:function(){return eQ.J$2},Less:function(){return eQ.vtC},LessEqual:function(){return eQ.CAk},LinSpace:function(){return eQ.e7N},Log:function(){return eQ.ZbH},Log1p:function(){return eQ.kU},LogSoftmax:function(){return eQ.qCd},LogicalAnd:function(){return eQ.PYm},LogicalNot:function(){return eQ.VfG},LogicalOr:function(){return eQ.MZg},LogicalXor:function(){return eQ.w6g},LowerBound:function(){return eQ.qIC},Max:function(){return eQ.YoZ},MaxPool:function(){return eQ.mTV},MaxPool3D:function(){return eQ.OAf},MaxPool3DGrad:function(){return eQ.OU7},MaxPoolGrad:function(){return eQ.OV7},MaxPoolWithArgmax:function(){return eQ.vFR},Maximum:function(){return eQ.BMI},Mean:function(){return eQ.q2K},Min:function(){return eQ.c17},Minimum:function(){return eQ.q8u},MirrorPad:function(){return eQ.jQs},Mod:function(){return eQ.Vbg},MomentumOptimizer:function(){return tY},Multinomial:function(){return eQ.NZg},Multiply:function(){return eQ.wYn},Neg:function(){return eQ.kuV},NonMaxSuppressionV3:function(){return eQ.uv1},NonMaxSuppressionV4:function(){return eQ.cye},NonMaxSuppressionV5:function(){return eQ.W0H},NotEqual:function(){return eQ.yQU},OP_SCOPE_SUFFIX:function(){return t2.zvA},OneHot:function(){return eQ.we_},OnesLike:function(){return eQ.qWM},Optimizer:function(){return tz},OptimizerConstructors:function(){return t0},Pack:function(){return eQ.QiL},PadV2:function(){return eQ.lyA},Pool:function(){return eQ.Kgp},Pow:function(){return eQ.pe_},Prelu:function(){return eQ.o0g},Prod:function(){return eQ.DlI},RMSPropOptimizer:function(){return tJ},RaggedGather:function(){return eQ.dDz},RaggedRange:function(){return eQ.CQl},RaggedTensorToTensor:function(){return eQ.BiW},Range:function(){return eQ.e6w},Rank:function(){return t1.yw},Real:function(){return eQ.xJR},RealDiv:function(){return eQ.oHH},Reciprocal:function(){return eQ.$HU},Reduction:function(){return t3.I},Relu:function(){return eQ.qkr},Relu6:function(){return eQ.SbG},Reshape:function(){return eQ.HZH},ResizeBilinear:function(){return eQ._Yw},ResizeBilinearGrad:function(){return eQ.zbQ},ResizeNearestNeighbor:function(){return eQ.dpD},ResizeNearestNeighborGrad:function(){return eQ.Hmb},Reverse:function(){return eQ.mKl},RotateWithOffset:function(){return eQ.b9H},Round:function(){return eQ.e07},Rsqrt:function(){return eQ.bV0},SGDOptimizer:function(){return tQ},ScatterNd:function(){return eQ.xQA},SearchSorted:function(){return eQ.nr8},Select:function(){return eQ.PhF},Selu:function(){return eQ.oFR},Sigmoid:function(){return eQ.a5O},Sign:function(){return eQ.i5y},Sin:function(){return eQ.RQH},Sinh:function(){return eQ.wYB},Slice:function(){return eQ.p2w},Softmax:function(){return eQ.Gcp},Softplus:function(){return eQ.MRv},SpaceToBatchND:function(){return eQ.TQc},SparseFillEmptyRows:function(){return eQ.O3z},SparseReshape:function(){return eQ.nhH},SparseSegmentMean:function(){return eQ.w3H},SparseSegmentSum:function(){return eQ.ZjV},SparseToDense:function(){return eQ.D2d},SplitV:function(){return eQ.L8s},Sqrt:function(){return eQ.FKq},Square:function(){return eQ.bK0},SquaredDifference:function(){return eQ._tC},Step:function(){return eQ.h8e},StridedSlice:function(){return eQ.jQk},StringNGrams:function(){return eQ._JP},StringSplit:function(){return eQ.s1s},StringToHashBucketFast:function(){return eQ.XkS},Sub:function(){return eQ.Tr8},Sum:function(){return eQ.GBy},Tan:function(){return eQ.sEM},Tanh:function(){return eQ.MIZ},Tensor:function(){return ev.es},TensorBuffer:function(){return ev.YD},Tile:function(){return eQ.n9L},TopK:function(){return eQ.cWu},Transform:function(){return eQ.wx7},Transpose:function(){return eQ.G3Y},Unique:function(){return eQ.kpP},Unpack:function(){return eQ.ToN},UnsortedSegmentSum:function(){return eQ.Qvg},UpperBound:function(){return eQ.XDQ},Variable:function(){return ev._w},ZerosLike:function(){return eQ.RuY},_FusedMatMul:function(){return eQ.usg},abs:function(){return t2.WnP},acos:function(){return t2.Khb},acosh:function(){return t2.__u},add:function(){return t2.IHx},addN:function(){return t2.QBD},all:function(){return t2.$6P},any:function(){return t2.YjB},argMax:function(){return t2.NqF},argMin:function(){return t2.vHJ},asin:function(){return t2.ZRM},asinh:function(){return t2.VfV},atan:function(){return t2.z4N},atan2:function(){return t2.fvJ},atanh:function(){return t2.C80},avgPool:function(){return t2.wS1},avgPool3d:function(){return t2.uR5},backend:function(){return t$.y3},backend_util:function(){return d},basicLSTMCell:function(){return t2.zEQ},batchNorm:function(){return t2.tgs},batchNorm2d:function(){return t2.Dxk},batchNorm3d:function(){return t2.JY5},batchNorm4d:function(){return t2.p3b},batchToSpaceND:function(){return t2.E4h},bincount:function(){return t2.yE8},booleanMaskAsync:function(){return t2.anm},broadcastArgs:function(){return t2.XsQ},broadcastTo:function(){return t2.UFq},broadcast_util:function(){return eZ},browser:function(){return o},buffer:function(){return t2.f3b},cast:function(){return t2.pju},ceil:function(){return t2.mDi},clipByValue:function(){return t2.iUl},clone:function(){return t2.d9v},complex:function(){return t2.PYB},concat:function(){return t2.zoF},concat1d:function(){return t2.gME},concat2d:function(){return t2.Izb},concat3d:function(){return t2.MNy},concat4d:function(){return t2.ZaL},conv1d:function(){return t2.PAt},conv2d:function(){return t2.Tek},conv2dTranspose:function(){return t2.bc},conv3d:function(){return t2.pdZ},conv3dTranspose:function(){return t2.$QV},copyRegisteredKernels:function(){return eY.T3},cos:function(){return t2.mCk},cosh:function(){return t2.f9Y},cosineWindow:function(){return t2.mew},cumprod:function(){return t2.$Gn},cumsum:function(){return t2.zbp},customGrad:function(){return tP.cb},denseBincount:function(){return t2.ppE},deprecationWarn:function(){return t$.MX},depthToSpace:function(){return t2.nTT},depthwiseConv2d:function(){return t2.B10},device_util:function(){return a},diag:function(){return t2.Ka3},dilation2d:function(){return t2.WmZ},disableDeprecationWarnings:function(){return t$.cF},dispose:function(){return t$.B9},disposeVariables:function(){return t$.N8},div:function(){return t2.hiC},divNoNan:function(){return t2.NTj},dot:function(){return t2.AKD},dropout:function(){return t2.rvX},einsum:function(){return t2.WYO},elu:function(){return t2.pyx},enableDebugMode:function(){return t$.R},enableProdMode:function(){return t$.G4},enclosingPowerOfTwo:function(){return t2.GRh},engine:function(){return t$.SR},env:function(){return N.OB},equal:function(){return t2.DgJ},erf:function(){return t2.qNN},euclideanNorm:function(){return t2.d2q},exp:function(){return t2.Qqt},expandDims:function(){return t2.dt4},expm1:function(){return t2.t$B},eye:function(){return t2.iyy},fft:function(){return t2.kp_},fill:function(){return t2.hlL},findBackend:function(){return t$.x3},findBackendFactory:function(){return t$.ze},floor:function(){return t2.GWj},floorDiv:function(){return t2.qPi},fused:function(){return t2.imm},gather:function(){return t2.Iqj},gatherND:function(){return t2.dbB},gather_util:function(){return u},getBackend:function(){return t$.N_},getGradient:function(){return eY.uk},getKernel:function(){return eY.pI},getKernelsForBackend:function(){return eY.tr},grad:function(){return tP.UQ},grads:function(){return tP.ti},greater:function(){return t2.pjt},greaterEqual:function(){return t2.brS},ifft:function(){return t2.Sxn},imag:function(){return t2.asL},image:function(){return t2.BHj},inTopKAsync:function(){return t2.V3u},io:function(){return s},irfft:function(){return t2.wx0},isFinite:function(){return t2.xVT},isInf:function(){return t2.UWc},isNaN:function(){return t2.i2d},keep:function(){return t$.Cn},kernel_impls:function(){return f},leakyRelu:function(){return t2.hi7},less:function(){return t2.d9m},lessEqual:function(){return t2.zN1},linalg:function(){return t2.$r2},linspace:function(){return t2.SX3},localResponseNormalization:function(){return t2.G9k},log:function(){return t2.cM7},log1p:function(){return t2.Krr},logSigmoid:function(){return t2.e_t},logSoftmax:function(){return t2.CmS},logSumExp:function(){return t2.l_t},logicalAnd:function(){return t2.HvI},logicalNot:function(){return t2.hJK},logicalOr:function(){return t2.K5V},logicalXor:function(){return t2.egP},losses:function(){return t2.MB5},lowerBound:function(){return t2.eab},matMul:function(){return t2.OI3},math:function(){return i},max:function(){return t2.Fp7},maxPool:function(){return t2._sB},maxPool3d:function(){return t2.YQQ},maxPoolWithArgmax:function(){return t2.Ip$},maximum:function(){return t2.gWQ},mean:function(){return t2.J69},memory:function(){return t$.sq},meshgrid:function(){return t2.ry_},min:function(){return t2.VV$},minimum:function(){return t2.LTh},mirrorPad:function(){return t2.VdP},mod:function(){return t2.wQq},moments:function(){return t2.Gi7},movingAverage:function(){return t2.p_},mul:function(){return t2.dC7},multiRNNCell:function(){return t2.rq4},multinomial:function(){return t2.SJ_},neg:function(){return t2.W76},nextFrame:function(){return t5},norm:function(){return t2.KOy},notEqual:function(){return t2.Quu},oneHot:function(){return t2.lfX},ones:function(){return t2.iUs},onesLike:function(){return t2.JpU},op:function(){return t2.op},outerProduct:function(){return t2.N2O},pad:function(){return t2.vku},pad1d:function(){return t2.pNR},pad2d:function(){return t2.koy},pad3d:function(){return t2.t1L},pad4d:function(){return t2.lGY},pool:function(){return t2.d_R},pow:function(){return t2.sQ3},prelu:function(){return t2.AL3},print:function(){return t2.S0v},prod:function(){return t2.WVs},profile:function(){return t$.N5},raggedGather:function(){return t2.$gW},raggedRange:function(){return t2.VT$},raggedTensorToTensor:function(){return t2.N89},rand:function(){return t2.TN_},randomGamma:function(){return t2.wzB},randomNormal:function(){return t2.nGf},randomStandardNormal:function(){return t2.ruB},randomUniform:function(){return t2.LGj},range:function(){return t2.w6H},ready:function(){return t$.Cd},real:function(){return t2.kwC},reciprocal:function(){return t2.M25},registerBackend:function(){return t$.jq},registerGradient:function(){return eY.Li},registerKernel:function(){return eY.wC},relu:function(){return t2.UYe},relu6:function(){return t2.btT},removeBackend:function(){return t$.cj},reshape:function(){return t2.XLQ},reverse:function(){return t2.GYS},reverse1d:function(){return t2.SDf},reverse2d:function(){return t2.diP},reverse3d:function(){return t2.sx7},reverse4d:function(){return t2.mG2},rfft:function(){return t2.QEs},round:function(){return t2.NMM},rsqrt:function(){return t2.bp0},scalar:function(){return t2.iD$},scatterND:function(){return t2.snQ},scatter_util:function(){return e5},searchSorted:function(){return t2.zcT},selu:function(){return t2.U8D},separableConv2d:function(){return t2.U_I},serialization:function(){return p},setBackend:function(){return t$.CQ},setPlatform:function(){return t$.VY},setdiff1dAsync:function(){return t2.ODp},sigmoid:function(){return t2.XD2},sign:function(){return t2.Xxe},signal:function(){return t2.tdS},sin:function(){return t2.O$l},sinh:function(){return t2.R_K},slice:function(){return t2.tPi},slice1d:function(){return t2.jZU},slice2d:function(){return t2.SmN},slice3d:function(){return t2.CnO},slice4d:function(){return t2.p0P},slice_util:function(){return l},softmax:function(){return t2.XAC},softplus:function(){return t2.Wvh},spaceToBatchND:function(){return t2.fBT},sparse:function(){return t2.rVs},sparseToDense:function(){return t2.ers},spectral:function(){return t2.uN7},split:function(){return t2.Vl2},sqrt:function(){return t2._b3},square:function(){return t2.h62},squaredDifference:function(){return t2.$i},squeeze:function(){return t2.L9e},stack:function(){return t2.knu},step:function(){return t2.Nbs},stridedSlice:function(){return t2.NXj},string:function(){return t2.Z_8},sub:function(){return t2.luU},sum:function(){return t2.Smz},sumOutType:function(){return t1.z4},tan:function(){return t2.ORZ},tanh:function(){return t2.AEp},tensor:function(){return t2.XeE},tensor1d:function(){return t2.RRF},tensor2d:function(){return t2.odF},tensor3d:function(){return t2.wOQ},tensor4d:function(){return t2.yXz},tensor5d:function(){return t2.Bfx},tensor6d:function(){return t2.xZs},tensor_util:function(){return ty},test_util:function(){return c},tidy:function(){return t$.lu},tile:function(){return t2.Gg6},time:function(){return t$.XV},topk:function(){return t2.hg7},train:function(){return t6},transpose:function(){return t2.p4s},truncatedNormal:function(){return t2.Xu6},unique:function(){return t2.Two},unregisterGradient:function(){return eY.bt},unregisterKernel:function(){return eY.nE},unsortedSegmentSum:function(){return t2.pUJ},unstack:function(){return t2.HHK},upcastType:function(){return t1.x8},upperBound:function(){return t2.GaM},util:function(){return tb},valueAndGrad:function(){return tP.h7},valueAndGrads:function(){return tP.fN},variable:function(){return t2.VD$},variableGrads:function(){return tP.pn},version_core:function(){return tD},where:function(){return t2.arb},whereAsync:function(){return t2.itS},zeros:function(){return t2.lls},zerosLike:function(){return t2.P84}});var r,a={};n.r(a),n.d(a,{isBrowser:function(){return k},isMobile:function(){return b},mockIsMobile:function(){return y}});var s={};n.r(s),n.d(s,{browserFiles:function(){return e_},browserHTTPRequest:function(){return eR},concatenateArrayBuffers:function(){return $},copyModel:function(){return ep},decodeWeights:function(){return A},encodeWeights:function(){return E},fromMemory:function(){return eL},fromMemorySync:function(){return ez},getLoadHandlers:function(){return G},getModelArtifactsForJSON:function(){return R},getModelArtifactsForJSONSync:function(){return O},getModelArtifactsInfoForJSON:function(){return C},getSaveHandlers:function(){return U},getWeightSpecs:function(){return V},http:function(){return eO},isHTTPScheme:function(){return eF},listModels:function(){return eu},loadWeights:function(){return eM},moveModel:function(){return ec},registerLoadRouter:function(){return W},registerSaveRouter:function(){return z},removeModel:function(){return el},weightsLoaderFactory:function(){return eD},withSaveHandler:function(){return eW},withSaveHandlerSync:function(){return eU}});var i={};n.r(i),n.d(i,{confusionMatrix:function(){return eX}});var o={};n.r(o),n.d(o,{fromPixels:function(){return e6},fromPixelsAsync:function(){return e2},toPixels:function(){return e3}});var u={};n.r(u),n.d(u,{prepareAndValidate:function(){return e4}});var l={};n.r(l),n.d(l,{assertParamsValid:function(){return e8},computeFlatOffset:function(){return tp},computeOutShape:function(){return e9},getNormalizedAxes:function(){return tr},isSliceContinous:function(){return tl},maskToAxes:function(){return e7},parseSliceParams:function(){return tc},sliceInfo:function(){return th},startForAxis:function(){return to},startIndicesWithElidedDims:function(){return ta},stopForAxis:function(){return tu},stopIndicesWithElidedDims:function(){return ts},stridesForAxis:function(){return ti},stridesWithElidedDims:function(){return te}});var p={};n.r(p),n.d(p,{Serializable:function(){return tf},SerializationMap:function(){return tm},registerClass:function(){return tg}});var c={};n.r(c),n.d(c,{TEST_EPSILON_FLOAT16:function(){return tk},createVideoElement:function(){return tA},encodeStrings:function(){return function e(t){for(let n=0;n<t.length;n++){let r=t[n];Array.isArray(r)?e(r):t[n]=(0,tb.encodeString)(r)}return t}},expectArrayBuffersEqual:function(){return tE},expectArraysClose:function(){return tN},expectArraysEqual:function(){return tT},expectNumbersClose:function(){return tS},expectPromiseToFail:function(){return tw},expectValuesInRange:function(){return t_},play:function(){return tM},testEpsilon:function(){return tv}});var h={};n.r(h),n.d(h,{collectGatherOpShapeInfo:function(){return n0},computeOutShape:function(){return nJ},segOpComputeOptimalWindowSize:function(){return nY}});var d={};n.r(d),n.d(d,{ERF_A1:function(){return nb},ERF_A2:function(){return nk},ERF_A3:function(){return nN},ERF_A4:function(){return nv},ERF_A5:function(){return nx},ERF_P:function(){return ny},PARALLELIZE_THRESHOLD:function(){return no},RowPartitionType:function(){return r},SELU_SCALE:function(){return ng},SELU_SCALEALPHA:function(){return nm},applyActivation:function(){return nn.QH},assertAndGetBroadcastShape:function(){return eZ.assertAndGetBroadcastShape},assertAxesAreInnerMostDims:function(){return t8.lB},assertParamsConsistent:function(){return t7},assignToTypedArray:function(){return nA},axesAreInnerMostDims:function(){return t8.YB},calculateShapes:function(){return e5.calculateShapes},checkEinsumDimSizes:function(){return nO},checkPadOnDimRoundingMode:function(){return nt.m},combineLocations:function(){return t8.Vh},combineRaggedTensorToTensorShapes:function(){return nr},complexWithEvenIndex:function(){return nI},complexWithOddIndex:function(){return n_},computeConv2DInfo:function(){return nt.Ix},computeConv3DInfo:function(){return nt.jw},computeDefaultPad:function(){return nt.aO},computeDilation2DInfo:function(){return nt.Rf},computeOptimalWindowSize:function(){return nu},computeOutAndReduceShapes:function(){return t8.kz},computeOutShape:function(){return t9},computePool2DInfo:function(){return nt.Xw},computePool3DInfo:function(){return nt.pl},convertConv2DDataFormat:function(){return nt.sl},decodeEinsumEquation:function(){return nF},eitherStridesOrDilationsAreOne:function(){return nt.jT},expandShapeToKeepDim:function(){return t8.rv},exponent:function(){return nD},exponents:function(){return nM},fromStringArrayToUint8:function(){return n2},fromUint8ToStringArray:function(){return n1},getAxesPermutation:function(){return t8.Q3},getBroadcastDims:function(){return eZ.getBroadcastDims},getComplexWithIndex:function(){return nE},getEinsumComputePath:function(){return nR},getEinsumPermutation:function(){return nB},getFusedBiasGradient:function(){return nn.pf},getFusedDyActivation:function(){return nn.Fr},getImageCenter:function(){return nl},getInnerMostAxes:function(){return t8.sY},getPermuted:function(){return nc},getRaggedRank:function(){return ns},getReductionAxes:function(){return eZ.getReductionAxes},getReshaped:function(){return np},getReshapedPermuted:function(){return nh},getRowPartitionTypesHelper:function(){return na},getSliceBeginCoords:function(){return nd},getSliceSize:function(){return nf},getSparseFillEmptyRowsIndicesDenseShapeMismatch:function(){return nL},getSparseFillEmptyRowsNegativeIndexErrorMessage:function(){return nz},getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:function(){return nW},getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:function(){return nq},getSparseReshapeInputOutputMismatchErrorMessage:function(){return nj},getSparseReshapeInputOutputMultipleErrorMessage:function(){return nH},getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:function(){return nU},getSparseReshapeNegativeOutputDimErrorMessage:function(){return nG},getSparseSegmentReductionIndicesOutOfRangeErrorMessage:function(){return nQ},getSparseSegmentReductionNegativeSegmentIdsErrorMessage:function(){return nK},getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:function(){return nX},getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:function(){return nZ},getUndoAxesPermutation:function(){return t8.LJ},isIdentityPermutation:function(){return nC},log:function(){return nw.c},mergeRealAndImagArrays:function(){return nT},prepareAndValidate:function(){return e4},prepareSplitSize:function(){return nP},segment_util:function(){return h},shouldFuse:function(){return nn.uy},slice_util:function(){return l},splitRealAndImagArrays:function(){return nS},tupleValuesAreOne:function(){return nt.I0},upcastType:function(){return t1.x8},validateDefaultValueShape:function(){return ni},validateInput:function(){return e5.validateInput},validateUpdateShape:function(){return e5.validateUpdateShape},warn:function(){return nw.Z}});var f={};n.r(f),n.d(f,{nonMaxSuppressionV3Impl:function(){return n3.GP},nonMaxSuppressionV4Impl:function(){return n3.qP},nonMaxSuppressionV5Impl:function(){return n3.pA},whereImpl:function(){return n6.Z}});var m=n(196);let g;function y(e){g=e}function b(e){if(void 0!==g)return g;if(e||"undefined"!=typeof navigator&&null!=navigator){if(e||(e=navigator),"ReactNative"===e.product)return!0;let t=e.userAgent||e.vendor||("undefined"!=typeof window?window.opera:"");if(!t){let n=e;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function k(){return"undefined"!=typeof window&&null!=window.document||"undefined"!=typeof WorkerGlobalScope}var N=n(2885),v=n(3454);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let x=(0,N.OB)();x.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")}),x.registerFlag("IS_BROWSER",()=>k()),x.registerFlag("IS_NODE",()=>void 0!==v&&void 0!==v.versions&&void 0!==v.versions.node),x.registerFlag("IS_CHROME",()=>"undefined"!=typeof navigator&&null!=navigator&&null!=navigator.userAgent&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),x.registerFlag("PROD",()=>!1),x.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>x.getBool("DEBUG")),x.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0),x.registerFlag("IS_TEST",()=>!1),x.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>!0),x.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1),x.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1),x.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);var w=n(1661),T=n(701),S=n(569);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let I={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};var _=n(1876).Buffer;async function E(e,t){let n=[],r=[],a=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);for(let s=0;s<a.length;++s){let i=a[s],o=Array.isArray(e)?e[s].tensor:e[i];if("float32"!==o.dtype&&"int32"!==o.dtype&&"bool"!==o.dtype&&"string"!==o.dtype&&"complex64"!==o.dtype)throw Error(`Unsupported dtype in weight '${i}': ${o.dtype}`);let u={name:i,shape:o.shape,dtype:o.dtype};if("string"===o.dtype){let l=new Promise(async e=>{let t=await o.bytes(),n=t.reduce((e,t)=>e+t.length,0)+4*t.length,r=new Uint8Array(n),a=0;for(let s=0;s<t.length;s++){let i=t[s],u=new Uint8Array(new Uint32Array([i.length]).buffer);r.set(u,a),a+=4,r.set(i,a),a+=i.length}e(r)});r.push(l)}else r.push(o.data());null!=t&&(u.group=t),n.push(u)}let p=await Promise.all(r);return{data:function(e){if(null===e)throw Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0,n=[];e.forEach(e=>{if(t+=e.byteLength,n.push(e.byteLength===e.buffer.byteLength?e:new e.constructor(e)),!(e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array))throw Error(`Unsupported TypedArray subtype: ${e.constructor.name}`)});let r=new Uint8Array(t),a=0;return n.forEach(e=>{r.set(new Uint8Array(e.buffer),a),a+=e.byteLength}),r.buffer}(p),specs:n}}function A(e,t){let n={},r,a=0;for(let s of t){let i=s.name,o=s.dtype,u=s.shape,l=(0,S.NA)(u),p;if("quantization"in s){let c=s.quantization;if("uint8"===c.dtype||"uint16"===c.dtype){if(!("min"in c&&"scale"in c))throw Error(`Weight ${s.name} with quantization ${c.dtype} doesn't have corresponding metadata min and scale.`)}else if("float16"===c.dtype){if("float32"!==o)throw Error(`Weight ${s.name} is quantized with ${c.dtype} which only supports weights of type float32 not ${o}.`)}else throw Error(`Weight ${s.name} has unknown quantization dtype ${c.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);let h=I[c.dtype],d=e.slice(a,a+l*h),f="uint8"===c.dtype?new Uint8Array(d):new Uint16Array(d);if("float32"===o){if("uint8"===c.dtype||"uint16"===c.dtype){p=new Float32Array(f.length);for(let m=0;m<f.length;m++){let g=f[m];p[m]=g*c.scale+c.min}}else if("float16"===c.dtype)void 0===r&&(r=P()),p=r(f);else throw Error(`Unsupported quantization type ${c.dtype} for weight type float32.`)}else if("int32"===o){if("uint8"!==c.dtype&&"uint16"!==c.dtype)throw Error(`Unsupported quantization type ${c.dtype} for weight type int32.`);p=new Int32Array(f.length);for(let y=0;y<f.length;y++){let b=f[y];p[y]=Math.round(b*c.scale+c.min)}}else throw Error(`Unsupported dtype in weight '${i}': ${o}`);a+=l*h}else if("string"===o){let k=(0,S.NA)(s.shape);p=[];for(let N=0;N<k;N++){let v=new Uint32Array(e.slice(a,a+4))[0];a+=4;let x=new Uint8Array(e.slice(a,a+v));p.push(x),a+=v}}else{let _=I[o],E=e.slice(a,a+l*_);if("float32"===o)p=new Float32Array(E);else if("int32"===o)p=new Int32Array(E);else if("bool"===o)p=new Uint8Array(E);else if("complex64"===o){p=new Float32Array(E);let A=new Float32Array(p.length/2),M=new Float32Array(p.length/2);for(let D=0;D<A.length;D++)A[D]=p[2*D],M[D]=p[2*D+1];let $=(0,T.X)(A,u,"float32"),F=(0,T.X)(M,u,"float32");n[i]=(0,w.P)($,F),$.dispose(),F.dispose()}else throw Error(`Unsupported dtype in weight '${i}': ${o}`);a+=l*_}"complex64"!==o&&(n[i]=(0,T.X)(p,u,o))}return n}let M=void 0!==_&&("undefined"==typeof Blob||"undefined"==typeof atob||"undefined"==typeof btoa);function D(e){return M?_.byteLength(e):new Blob([e]).size}function $(e){if(1===e.length)return e[0];let t=0;e.forEach(e=>{t+=e.byteLength});let n=new Uint8Array(t),r=0;return e.forEach(e=>{n.set(new Uint8Array(e),r),r+=e.byteLength}),n.buffer}function F(e){for(e=e.trim();e.endsWith("/");)e=e.slice(0,e.length-1);let t=e.split("/");return t[t.length-1]}function B(e,t){let n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return null!=e.signature&&(n.signature=e.signature),null!=e.userDefinedMetadata&&(n.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(n.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(n.initializerSignature=e.initializerSignature),null!=e.trainingConfig&&(n.trainingConfig=e.trainingConfig),n}function O(e,t,n){let r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(null!=e.trainingConfig&&(r.trainingConfig=e.trainingConfig),null!=e.weightsManifest){if(!t)throw Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw Error("modelJSON has weightsManifest but weightData is null");r.weightSpecs=t,r.weightData=n}return null!=e.signature&&(r.signature=e.signature),null!=e.userDefinedMetadata&&(r.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(r.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(r.initializerSignature=e.initializerSignature),r}async function R(e,t){let n,r;return null!=e.weightsManifest&&([n,r]=await t(e.weightsManifest)),O(e,n,r)}function C(e){if(e.modelTopology instanceof ArrayBuffer)throw Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:null==e.modelTopology?0:D(JSON.stringify(e.modelTopology)),weightSpecsBytes:null==e.weightSpecs?0:D(JSON.stringify(e.weightSpecs)),weightDataBytes:null==e.weightData?0:e.weightData.byteLength}}function V(e){let t=[];for(let n of e)t.push(...n.weights);return t}function P(){let e=function(){let e=e=>{let t=e<<13,n=0;for(;(8388608&t)==0;)n-=8388608,t<<=1;return(t&=-8388609)|(n+=947912704)},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let r=1024;r<2048;r++)t[r]=939524096+(r-1024<<13);return t}(),t=function(){let e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let n=33;n<63;n++)e[n]=2147483648+(n-32<<23);return e}(),n=function(){let e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}();return r=>{let a=new ArrayBuffer(4*r.length),s=new Uint32Array(a);for(let i=0;i<r.length;i++){let o=r[i],u=e[n[o>>10]+(1023&o)]+t[o>>10];s[i]=u}return new Float32Array(a)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class L{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return null==L.instance&&(L.instance=new L),L.instance}static registerSaveRouter(e){L.getInstance().saveRouters.push(e)}static registerLoadRouter(e){L.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return L.getHandlers(e,"save")}static getLoadHandlers(e,t){return L.getHandlers(e,"load",t)}static getHandlers(e,t,n){let r=[],a="load"===t?L.getInstance().loadRouters:L.getInstance().saveRouters;return a.forEach(t=>{let a=t(e,n);null!==a&&r.push(a)}),r}}let z=e=>L.registerSaveRouter(e),W=e=>L.registerLoadRouter(e),U=e=>L.getSaveHandlers(e),G=(e,t)=>L.getLoadHandlers(e,t),q="tensorflowjs",H="models_store",j="model_info_store";function K(){if(!(0,N.OB)().getBool("IS_BROWSER"))throw Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");let e="undefined"==typeof window?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(null==t)throw Error("The current browser does not appear to support IndexedDB.");return t}function X(e){let t=e.result;t.createObjectStore(H,{keyPath:"modelPath"}),t.createObjectStore(j,{keyPath:"modelPath"})}class Z{constructor(e){if(this.indexedDB=K(),null==e||!e)throw Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{let r=this.indexedDB.open(q,1);r.onupgradeneeded=()=>X(r),r.onsuccess=()=>{let a=r.result;if(null==t){let s=a.transaction(H,"readonly"),i=s.objectStore(H),o=i.get(this.modelPath);o.onsuccess=()=>{if(null==o.result)return a.close(),n(Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(o.result.modelArtifacts)},o.onerror=e=>(a.close(),n(o.error)),s.oncomplete=()=>a.close()}else{let u=C(t),l=a.transaction(j,"readwrite"),p=l.objectStore(j),c=p.put({modelPath:this.modelPath,modelArtifactsInfo:u}),h;c.onsuccess=()=>{h=a.transaction(H,"readwrite");let r=h.objectStore(H),s=r.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:u});s.onsuccess=()=>e({modelArtifactsInfo:u}),s.onerror=e=>{p=l.objectStore(j);let t=p.delete(this.modelPath);t.onsuccess=()=>(a.close(),n(s.error)),t.onerror=e=>(a.close(),n(s.error))}},c.onerror=e=>(a.close(),n(c.error)),l.oncomplete=()=>{null==h?a.close():h.oncomplete=()=>a.close()}}},r.onerror=e=>n(r.error)})}}Z.URL_SCHEME="indexeddb://";let Q=e=>{var t;return(0,N.OB)().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Z.URL_SCHEME)?(t=e.slice(Z.URL_SCHEME.length),new Z(t)):null};L.registerSaveRouter(Q),L.registerLoadRouter(Q);let Y="tensorflowjs_models",J="info";function ee(e){return{info:[Y,e,J].join("/"),topology:[Y,e,"model_topology"].join("/"),weightSpecs:[Y,e,"weight_specs"].join("/"),weightData:[Y,e,"weight_data"].join("/"),modelMetadata:[Y,e,"model_metadata"].join("/")}}function et(e){for(let t of Object.values(e))window.localStorage.removeItem(t)}function en(e){let t=e.split("/");if(t.length<3)throw Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join("/")}class er{constructor(e){if(!(0,N.OB)().getBool("IS_BROWSER")||"undefined"==typeof window||void 0===window.localStorage)throw Error("The current environment does not support local storage.");if(this.LS=window.localStorage,null==e||!e)throw Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=ee(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{let t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=C(e);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,function(e){if(M)return _.from(e).toString("base64");let t=new Uint8Array(e),n="";for(let r=0,a=t.length;r<a;r++)n+=String.fromCharCode(t[r]);return btoa(n)}(e.weightData));let a={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:null!=e.signature?e.signature:void 0,userDefinedMetadata:null!=e.userDefinedMetadata?e.userDefinedMetadata:void 0,modelInitializer:null!=e.modelInitializer?e.modelInitializer:void 0,initializerSignature:null!=e.initializerSignature?e.initializerSignature:void 0,trainingConfig:null!=e.trainingConfig?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch(s){throw et(this.keys),Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){let e=JSON.parse(this.LS.getItem(this.keys.info));if(null==e)throw Error(`In local storage, there is no model with name '${this.modelPath}'`);if("JSON"!==e.modelTopologyType)throw Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");let t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(null==n)throw Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;let r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(null==r)throw Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;let a=this.LS.getItem(this.keys.modelMetadata);if(null!=a){let s=JSON.parse(a);t.format=s.format,t.generatedBy=s.generatedBy,t.convertedBy=s.convertedBy,null!=s.signature&&(t.signature=s.signature),null!=s.userDefinedMetadata&&(t.userDefinedMetadata=s.userDefinedMetadata),null!=s.modelInitializer&&(t.modelInitializer=s.modelInitializer),null!=s.initializerSignature&&(t.initializerSignature=s.initializerSignature),null!=s.trainingConfig&&(t.trainingConfig=s.trainingConfig)}let i=this.LS.getItem(this.keys.weightData);if(null==i)throw Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=function(e){if(M){let t=_.from(e,"base64");return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}let n=atob(e),r=new Uint8Array(n.length);for(let a=0;a<n.length;++a)r.set([n.charCodeAt(a)],a);return r.buffer}(i),t}}er.URL_SCHEME="localstorage://";let ea=e=>{var t;return(0,N.OB)().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(er.URL_SCHEME)?(t=e.slice(er.URL_SCHEME.length),new er(t)):null};L.registerSaveRouter(ea),L.registerLoadRouter(ea);class es{constructor(){this.managers={}}static getInstance(){return null==es.instance&&(es.instance=new es),es.instance}static registerManager(e,t){(0,S.hu)(null!=e,()=>"scheme must not be undefined or null."),e.endsWith("://")&&(e=e.slice(0,e.indexOf("://"))),(0,S.hu)(e.length>0,()=>"scheme must not be an empty string.");let n=es.getInstance();(0,S.hu)(null==n.managers[e],()=>`A model store manager is already registered for scheme '${e}'.`),n.managers[e]=t}static getManager(e){let t=es.getInstance().managers[e];if(null==t)throw Error(`Cannot find model manager for scheme '${e}'`);return t}static getSchemes(){return Object.keys(es.getInstance().managers)}}function ei(e){if(-1===e.indexOf("://"))throw Error(`The url string provided does not contain a scheme. Supported schemes are: ${es.getSchemes().join(",")}`);return{scheme:e.split("://")[0],path:e.split("://")[1]}}async function eo(e,t,n=!1){(0,S.hu)(e!==t,()=>`Old path and new path are the same: '${e}'`);let r=L.getLoadHandlers(e);(0,S.hu)(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),(0,S.hu)(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);let a=r[0],s=L.getSaveHandlers(t);(0,S.hu)(s.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),(0,S.hu)(s.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);let i=s[0],o=ei(e).scheme,u=ei(e).path,l=o===ei(e).scheme,p=await a.load();n&&l&&await es.getManager(o).removeModel(u);let c=await i.save(p);return n&&!l&&await es.getManager(o).removeModel(u),c.modelArtifactsInfo}async function eu(){let e=es.getSchemes(),t={};for(let n of e){let r=await es.getManager(n).listModels();for(let a in r){let s=n+"://"+a;t[s]=r[a]}}return t}async function el(e){let t=ei(e),n=es.getManager(t.scheme);return n.removeModel(t.path)}async function ep(e,t){return eo(e,t,!1)}async function ec(e,t){return eo(e,t,!0)}if((0,N.OB)().get("IS_BROWSER")){(0,N.OB)().setPlatform("browser",new /**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if("utf-8"!==t&&"utf8"!==t)throw Error(`Browser's encoder only supports utf-8, but got ${t}`);return null==this.textEncoder&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if("undefined"==typeof window||!(0,N.OB)().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();let t=this.functionRefs[e.data.index];t(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}});try{es.registerManager(er.URL_SCHEME,new class{constructor(){(0,S.hu)((0,N.OB)().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),(0,S.hu)("undefined"==typeof window||void 0!==window.localStorage,()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){let e={},t=Y+"/",n="/"+J;for(let r=0;r<this.LS.length;++r){let a=this.LS.key(r);if(a.startsWith(t)&&a.endsWith(n)){let s=en(a);e[s]=JSON.parse(this.LS.getItem(a))}}return e}async removeModel(e){var t;e=(t=e).startsWith(er.URL_SCHEME)?t.slice(er.URL_SCHEME.length):t;let n=ee(e);if(null==this.LS.getItem(n.info))throw Error(`Cannot find model at path '${e}'`);let r=JSON.parse(this.LS.getItem(n.info));return et(n),r}})}catch(eh){}try{es.registerManager(Z.URL_SCHEME,new class{constructor(){this.indexedDB=K()}async listModels(){return new Promise((e,t)=>{let n=this.indexedDB.open(q,1);n.onupgradeneeded=()=>X(n),n.onsuccess=()=>{let r=n.result,a=r.transaction(j,"readonly"),s=a.objectStore(j),i=s.getAll();i.onsuccess=()=>{let t={};for(let n of i.result)t[n.modelPath]=n.modelArtifactsInfo;e(t)},i.onerror=e=>(r.close(),t(i.error)),a.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){var t;return e=(t=e).startsWith(Z.URL_SCHEME)?t.slice(Z.URL_SCHEME.length):t,new Promise((t,n)=>{let r=this.indexedDB.open(q,1);r.onupgradeneeded=()=>X(r),r.onsuccess=()=>{let a=r.result,s=a.transaction(j,"readwrite"),i=s.objectStore(j),o=i.get(e),u;o.onsuccess=()=>{if(null==o.result)return a.close(),n(Error(`Cannot find model with path '${e}' in IndexedDB.`));{let r=i.delete(e),s=()=>{u=a.transaction(H,"readwrite");let r=u.objectStore(H),s=r.delete(e);s.onsuccess=()=>t(o.result.modelArtifactsInfo),s.onerror=e=>n(o.error)};r.onsuccess=s,r.onerror=e=>(s(),a.close(),n(o.error))}},o.onerror=e=>(a.close(),n(o.error)),s.oncomplete=()=>{null==u?a.close():u.oncomplete=()=>a.close()}},r.onerror=e=>n(r.error)})}})}catch(ed){}}var ef=n(3454);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let em={importFetch:()=>n(5410)},eg;(0,N.OB)().get("IS_NODE")&&!(0,N.OB)().get("IS_BROWSER")&&(0,N.OB)().setPlatform("node",new class{constructor(){this.util=n(8628),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return null!=(0,N.OB)().global.fetch?(0,N.OB)().global.fetch(e,t):(null==eg&&(eg=em.importFetch()),eg(e,t))}now(){let e=ef.hrtime();return 1e3*e[0]+e[1]/1e6}encode(e,t){if("utf-8"!==t&&"utf8"!==t)throw Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return 0===e.length?"":new this.util.TextDecoder(t).decode(e)}});var ey=n(2657),eb=n(2271),ek=n(8723),eN=n(9798),ev=n(974);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ (0,m.wv)();let ex={buffer:ey.f,cast:eb.p,clone:ek.d,print:eN.S};function ew(e){return new Promise(e=>setTimeout(e)).then(e)}(0,ev.Vp)(ex);class eT{constructor(e){if(!(0,N.OB)().getBool("IS_BROWSER"))throw Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(eT.URL_SCHEME)&&(e=e.slice(eT.URL_SCHEME.length)),(null==e||0===e.length)&&(e="model"),this.modelJsonFileName=e+".json",this.weightDataFileName=e+".weights.bin"}async save(e){if("undefined"==typeof document)throw Error("Browser downloads are not supported in this environment since `document` is not present");let t=window.URL.createObjectURL(new Blob([e.weightData],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{let n=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],r=B(e,n),a=window.URL.createObjectURL(new Blob([JSON.stringify(r)],{type:"application/json"})),s=null==this.modelJsonAnchor?document.createElement("a"):this.modelJsonAnchor;if(s.download=this.modelJsonFileName,s.href=a,await ew(()=>s.dispatchEvent(new MouseEvent("click"))),null!=e.weightData){let i=null==this.weightDataAnchor?document.createElement("a"):this.weightDataAnchor;i.download=this.weightDataFileName,i.href=t,await ew(()=>i.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:C(e)}}}}eT.URL_SCHEME="downloads://";class eS{constructor(e){if(null==e||e.length<1)throw Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{let n=new FileReader;n.onload=n=>{let r=JSON.parse(n.target.result),a=r.modelTopology;if(null==a){t(Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}let s=r.weightsManifest;if(null==s){t(Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(0===this.weightsFiles.length){e({modelTopology:a});return}let i=R(r,e=>this.loadWeights(e));e(i)},n.onerror=e=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),n.readAsText(this.jsonFile)})}loadWeights(e){let t=[],n=[];for(let r of e)t.push(...r.weights),n.push(...r.paths);let a=this.checkManifestAndWeightFiles(e),s=n.map(e=>this.loadWeightsFile(e,a[e]));return Promise.all(s).then(e=>[t,$(e)])}loadWeightsFile(e,t){return new Promise((n,r)=>{let a=new FileReader;a.onload=e=>{let t=e.target.result;n(t)},a.onerror=t=>r(`Failed to weights data from file of path '${e}'.`),a.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){let t=[],n=this.weightsFiles.map(e=>F(e.name)),r={};for(let a of e)a.paths.forEach(e=>{let a=F(e);if(-1!==t.indexOf(a))throw Error(`Duplicate file basename found in weights manifest: '${a}'`);if(t.push(a),-1===n.indexOf(a))throw Error(`Weight file with basename '${a}' is not provided.`);r[e]=this.weightsFiles[n.indexOf(a)]});if(t.length!==this.weightsFiles.length)throw Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}let eI=e=>(0,N.OB)().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(eT.URL_SCHEME)?function(e="model"){return new eT(e)}(e.slice(eT.URL_SCHEME.length)):null;function e_(e){return new eS(e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function eE(e,t,n,r){var a,s,i;a=e,(0,S.hu)(null!=a&&Array.isArray(a)&&a.length>0,()=>"promises must be a none empty array"),s=n=null==n?0:n,i=r=null==r?1:r,(0,S.hu)(s>=0&&s<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${s}`),(0,S.hu)(i>=0&&i<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${i}`),(0,S.hu)(i>=s,()=>`startFraction must be no more than endFraction, but got startFraction ${s} and endFraction ${i}`);let o=0,u=a=>(a.then(a=>{let s=n+ ++o/e.length*(r-n);return t(s),a}),a);return Promise.all(e.map(u))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function eA(e,t){null==t&&(t={});let n=null==t.fetchFunc?(0,N.OB)().platform.fetch:t.fetchFunc,r=e.map(e=>n(e,t.requestInit,{isBinary:!0})),a=null==t.onProgress?await Promise.all(r):await eE(r,t.onProgress,0,.5),s=a.map(e=>e.arrayBuffer()),i=null==t.onProgress?await Promise.all(s):await eE(s,t.onProgress,.5,1);return i}async function eM(e,t="",n,r){let a=e=>eA(e,{requestInit:r}),s=eD(a);return s(e,t,n)}function eD(e){return async(t,n="",r)=>{let a=t.map(()=>!1),s={},i=null!=r?r.map(()=>!1):[],o=[];if(t.forEach((e,t)=>{let n=0;e.weights.forEach(e=>{let u="quantization"in e?e.quantization.dtype:e.dtype,l=I[u]*S.NA(e.shape),p=()=>{a[t]=!0,null==s[t]&&(s[t]=[]),s[t].push({manifestEntry:e,groupOffset:n,sizeBytes:l})};null!=r?r.forEach((t,n)=>{t===e.name&&(p(),i[n]=!0)}):p(),o.push(e.name),n+=l})}),!i.every(e=>e)){let u=r.filter((e,t)=>!i[t]);throw Error(`Could not find weights in manifest with names: ${u.join(", ")}. 
Manifest JSON has weights with names: ${o.join(", ")}.`)}let l=a.reduce((e,t,n)=>(t&&e.push(n),e),[]),p=[];l.forEach(e=>{t[e].paths.forEach(e=>{let t=n+(n.endsWith("/")?"":"/")+e;p.push(t)})});let c=await e(p),h={},d=0;return l.forEach(e=>{let n=t[e].paths.length,r=0;for(let a=0;a<n;a++)r+=c[d+a].byteLength;let i=new ArrayBuffer(r),o=new Uint8Array(i),u=0;for(let l=0;l<n;l++){let p=new Uint8Array(c[d+l]);o.set(p,u),u+=p.byteLength}let f=s[e];f.forEach(e=>{let t=i.slice(e.groupOffset,e.groupOffset+e.sizeBytes),n=A(t,[e.manifestEntry]);for(let r in n)h[r]=n[r]}),d+=n}),h}}L.registerSaveRouter(eI);class e${constructor(e,t){if(this.DEFAULT_METHOD="POST",null==t&&(t={}),this.weightPathPrefix=t.weightPathPrefix,this.onProgress=t.onProgress,this.weightUrlConverter=t.weightUrlConverter,null!=t.fetchFunc?((0,S.hu)("function"==typeof t.fetchFunc,()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc):this.fetch=(0,N.OB)().platform.fetch,(0,S.hu)(null!=e&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&(0,S.hu)(2===e.length,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,null!=t.requestInit&&null!=t.requestInit.body)throw Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=t.requestInit||{}}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");let t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;let n=[{paths:["./model.weights.bin"],weights:e.weightSpecs}],r=B(e,n);t.body.append("model.json",new Blob([JSON.stringify(r)],{type:"application/json"}),"model.json"),null!=e.weightData&&t.body.append("model.weights.bin",new Blob([e.weightData],{type:"application/octet-stream"}),"model.weights.bin");let a=await this.fetch(this.path,t);if(a.ok)return{modelArtifactsInfo:C(e),responses:[a]};throw Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${a.status}.`)}async load(){let e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch(r){let n=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?n+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":n+=" Please make sure the server is serving valid JSON for this request.",Error(n)}let a=t.modelTopology,s=t.weightsManifest;if(null==a&&null==s)throw Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return R(t,e=>this.loadWeights(e))}async loadWeights(e){let t=Array.isArray(this.path)?this.path[1]:this.path,[n,r]=function(e){let t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),r=e.substring(0,t),a=n>t?e.substring(n):"";return[r+"/",a]}(t),a=this.weightPathPrefix||n,s=V(e),i=[],o=[];for(let u of e)for(let l of u.paths)null!=this.weightUrlConverter?o.push(this.weightUrlConverter(l)):i.push(a+l+r);this.weightUrlConverter&&i.push(...await Promise.all(o));let p=await eA(i,{requestInit:this.requestInit,fetchFunc:this.fetch,onProgress:this.onProgress});return[s,$(p)]}}function eF(e){return null!=e.match(e$.URL_SCHEME_REGEX)}e$.URL_SCHEME_REGEX=/^https?:\/\//;let eB=(e,t)=>{if("undefined"==typeof fetch&&(null==t||null==t.fetchFunc));else{let n=!0;if(Array.isArray(e)?e.every(e=>eF(e)):eF(e))return eO(e,t)}return null};function eO(e,t){return new e$(e,t)}function eR(e,t){return eO(e,t)}L.registerSaveRouter(eB),L.registerLoadRouter(eB);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class eC{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class eV{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class eP{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function eL(e,t,n,r){let a=arguments;return new eP(ez(...a))}function ez(e,t,n,r){if(1!==arguments.length)return console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new eC({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:r});{let a=null!=e.modelTopology||null!=e.weightSpecs;return a?new eC(e):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new eC({modelTopology:e}))}}function eW(e){return new eV(e)}function eU(e){return new eV(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ var eG=n(3740),eq=n(8687),eH=n(6708),ej=n(2668),eK=n(9065);let eX=(0,ej.op)({confusionMatrix_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,eG._1)(e,"labels","confusionMatrix"),a=(0,eG._1)(t,"predictions","confusionMatrix");S.hu(null==n||n>0&&Number.isInteger(n),()=>`If provided, numClasses must be a positive integer, but got ${n}`),S.hu(1===r.rank,()=>`Expected the rank of labels to be 1, but got ${r.rank}`),S.hu(1===a.rank,()=>`Expected the rank of predictions to be 1, but got ${a.rank}`),S.hu(r.shape[0]===a.shape[0],()=>`Mismatch in the number of examples: ${r.shape[0]} vs. ${a.shape[0]}. Labels and predictions should have the same number of elements.`),S.hu(n>0&&Number.isInteger(n),()=>`numClasses is required to be a positive integer, but got ${n}`);let s=(0,eH.l)((0,eb.p)(r,"int32"),n),i=(0,eH.l)((0,eb.p)(a,"int32"),n),o=(0,eK.p)(s),u=(0,eq.O)(o,i);return(0,eb.p)(u,"int32")}});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ var eZ=n(2200),eQ=n(9121),eY=n(6151),eJ=n(9906);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let e0;function e1(e,t=3){if(t>4)throw Error("Cannot construct Tensor with more than 4 channels from pixels.");if(null==e)throw Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,r=!1,a=!1,s=!1,i=!1,o=!1;if(e.data instanceof Uint8Array)n=!0;else if("undefined"!=typeof ImageData&&e instanceof ImageData)r=!0;else if("undefined"!=typeof HTMLVideoElement&&e instanceof HTMLVideoElement)a=!0;else if("undefined"!=typeof HTMLImageElement&&e instanceof HTMLImageElement)s=!0;else if(null!=e.getContext)i=!0;else if("undefined"!=typeof ImageBitmap&&e instanceof ImageBitmap)o=!0;else throw Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);let u=(0,eY.pI)(eQ.eBW,m.BV.backendName);if(null!=u)return m.BV.runKernel(eQ.eBW,{pixels:e},{numChannels:t});let[l,p]=a?[e.videoWidth,e.videoHeight]:[e.width,e.height],c;if(i)c=e.getContext("2d").getImageData(0,0,l,p).data;else if(r||n)c=e.data;else if(s||a||o){if(null==e0){if("undefined"==typeof document){if("undefined"!=typeof OffscreenCanvas&&"undefined"!=typeof OffscreenCanvasRenderingContext2D)e0=new OffscreenCanvas(1,1).getContext("2d");else throw Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.")}else e0=document.createElement("canvas").getContext("2d",{willReadFrequently:!0})}e0.canvas.width=l,e0.canvas.height=p,e0.drawImage(e,0,0,l,p),c=e0.getImageData(0,0,l,p).data}let h;if(4===t)h=new Int32Array(c);else{let d=l*p;h=new Int32Array(d*t);for(let f=0;f<d;f++)for(let g=0;g<t;++g)h[f*t+g]=c[4*f+g]}return(0,eJ.w)(h,[p,l,t],"int32")}async function e2(e,t=3){var n,r,a;let s=null;if((0,N.OB)().getBool("WRAP_TO_IMAGEBITMAP")&&"undefined"!=typeof window&&"undefined"!=typeof ImageBitmap&&window.hasOwnProperty("createImageBitmap")&&!(e instanceof ImageBitmap)&&null!=e&&0!==e.width&&0!==e.height&&!(null!=e&&e.data instanceof Uint8Array)){let i;try{i=await createImageBitmap(e,{premultiplyAlpha:"none"})}catch(o){i=null}s=null!=i&&i.width===e.width&&i.height===e.height?i:e}else s=e;return e1(s,t)}async function e3(e,t){let n=(0,eG._1)(e,"img","toPixels");if(!(e instanceof ev.es)){let r=n;n=(0,eb.p)(r,"int32"),r.dispose()}if(2!==n.rank&&3!==n.rank)throw Error(`toPixels only supports rank 2 or 3 tensors, got rank ${n.rank}.`);let[a,s]=n.shape.slice(0,2),i=2===n.rank?1:n.shape[2];if(i>4||2===i)throw Error(`toPixels only supports depth of size 1, 3 or 4 but got ${i}`);if("float32"!==n.dtype&&"int32"!==n.dtype)throw Error(`Unsupported type for toPixels: ${n.dtype}. Please use float32 or int32 tensors.`);let o=await n.data(),u="float32"===n.dtype?255:1,l=new Uint8ClampedArray(s*a*4);for(let p=0;p<a*s;++p){let c=[0,0,0,255];for(let h=0;h<i;h++){let d=o[p*i+h];if("float32"===n.dtype){if(d<0||d>1)throw Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${d}.`)}else if("int32"===n.dtype&&(d<0||d>255))throw Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${d}.`);1===i?(c[0]=d*u,c[1]=d*u,c[2]=d*u):c[h]=d*u}let f=4*p;l[f+0]=Math.round(c[0]),l[f+1]=Math.round(c[1]),l[f+2]=Math.round(c[2]),l[f+3]=Math.round(c[3])}if(null!=t){t.width=s,t.height=a;let m=t.getContext("2d"),g=new ImageData(l,s,a);m.putImageData(g,0,0)}return n!==e&&n.dispose(),l}let e6=(0,ej.op)({fromPixels_:e1});function e4(e,t){let n=e.shape.length,r=t.shape.length;if(n<1)throw Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if("int32"!==t.dtype)throw Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(0===(0,S.NA)(e.shape))throw Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);let a=t.shape,s=a[a.length-1],i=1;for(let o=0;o<a.length-1;++o)i*=a[o];let u=e.shape,l=a.slice();l.pop();let p=1;for(let c=s;c<n;++c)p*=u[c],l.push(u[c]);let h=[...(0,S.e3)(e.shape).map(e=>e/p),1].slice(0,s);return[l,i,p,h]}var e5=n(3028);function e8(e,t,n){let r=e.shape.length;S.hu(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),S.hu(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let a=0;a<r;++a)S.hu(t[a]+n[a]<=e.shape[a],()=>`Error in slice${r}D: begin[${a}] + size[${a}] (${t[a]+n[a]}) would overflow input.shape[${a}] (${e.shape[a]})`)}function e7(e){let t=[],n=0;for(;e>0;)1&e&&t.push(n),e/=2,n++;return t}function e9(e,t,n){let r=[];for(let a=0;a<e.length;a++)r[a]=Math.ceil((t[a]-e[a])/n[a]);return r}function te(e,t,n,r){let a=[...e];for(let s=a.length;s<r.length;s++)a.push(1);for(let i=0;i<n;i++)0===i?a[t]=1:(a.splice(t,0,1),a.pop());return a}function tt(e,t,n){return n<=e?n:n-(t-1)}function tn(e,t){let n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function tr(e,t,n,r,a,s,i,o,u){let l=e.length,p=Array(l),c=Array(l),h=Array(l);if(t.length&&n>0){let d=t[0],f=n+1;p=ta(i,d,f,r,e),c=ts(o,d,f,a,e),h=te(s,d,f,e)}else for(let m=0;m<l;m++)p[m]=to(i,r,s,e,m,u),c[m]=tu(o,a,s,e,m,u),h[m]=ti(s,m,u);return{begin:p,end:c,strides:h}}function ta(e,t,n,r,a){let s=[...a],i=tn(n,t);for(let o=0;o<s.length;o++)if(i.indexOf(o)>-1)s[o]=0;else{let u=tt(t,n,o),l=r[u];e&1<<u&&(l=0),s[o]=l}return s}function ts(e,t,n,r,a){let s=[...a],i=tn(n,t);for(let o=0;o<s.length;o++)if(i.indexOf(o)>-1)s[o]=Number.MAX_SAFE_INTEGER;else{let u=tt(t,n,o),l=r[u];e&1<<u&&(l=Number.MAX_SAFE_INTEGER),s[o]=l}for(let p=0;p<s.length;p++){let c=a[p];s[p]<0&&(s[p]+=c),s[p]=S.uZ(0,s[p],a[p])}return s}function ti(e,t,n){let r=e[t];return(n&1<<t||null==r)&&(r=1),r}function to(e,t,n,r,a,s){let i=t[a],o=n[a]||1;(e&1<<a||s&1<<a||null==i)&&(i=o>0?Number.MIN_SAFE_INTEGER:Number.MAX_SAFE_INTEGER);let u=r[a];return i<0&&(i+=u),i=S.uZ(0,i,u-1)}function tu(e,t,n,r,a,s){let i=t[a],o=n[a]||1;(e&1<<a||s&1<<a||null==i)&&(i=o>0?Number.MAX_SAFE_INTEGER:Number.MIN_SAFE_INTEGER);let u=r[a];return i<0&&(i+=u),i=o>0?S.uZ(0,i,u):S.uZ(-1,i,u-1)}function tl(e,t,n){let r=n.length;for(let a=0;a<n.length;a++)if(n[a]>1){r=a;break}for(let s=r+1;s<n.length;s++)if(t[s]>0||n[s]!==e[s])return!1;return!0}function tp(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function tc(e,t,n){let r,a=e.shape.length;(r="number"==typeof t?[t,...Array(a-1).fill(0)]:t.length<a?t.concat(Array(a-t.length).fill(0)):t.slice()).forEach(e=>{S.hu(-1!==e,()=>"slice() does not support negative begin indexing.")});let s;return s=(s=null==n?Array(a).fill(-1):"number"==typeof n?[n,...Array(a-1).fill(-1)]:n.length<a?n.concat(Array(a-n.length).fill(-1)):n).map((t,n)=>t>=0?t:(S.hu(-1===t,()=>`Negative size values should be exactly -1 but got ${t} for the slice() size at index ${n}.`),e.shape[n]-r[n])),[r,s]}function th(e,t,n,r,a,s,i,o,u){let l;if(null==r?(l=Array(t.length)).fill(1):l=r,null!=i&&(i&i-1)!=0)throw Error("Multiple ellipses in slice is not allowed.");let p=!1,c={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:a,endMask:s,ellipsisMask:i,newAxisMask:o,shrinkAxisMask:u};for(let h=0;h<c.dims;h++)p&&(1<<h&o)!=0&&c.numAddAxisAfterEllipsis++,1<<h&i&&(p=!0);!p&&(c.ellipsisMask|=1<<c.dims,c.dims++);let d={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};!function(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=null!=e.begin,t.endValid=null!=e.end,t.begin=Array(t.dims),t.end=Array(t.dims),t.strides=Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){let a=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<a;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(-2),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);null!=e.begin&&(t.begin[n]=e.begin[r]),null!=e.end&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(-1),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}(c,d);let f=!0,m=!0,g=!0,y=[],b=[];for(let k=0;k<e.length;++k){if(0===d.strides[k])throw Error(`strides[${k}] must be non-zero`);let N=!!(d.shrinkAxisMask&1<<k),v=e[k];if(-1===v){y.push(N?1:-1);continue}let x=[d.beginMask&1<<k,d.endMask&1<<k],w=[d.strides[k]>0?0:-1,d.strides[k]>0?v:v-1];if(N&&d.strides[k]<=0)throw Error("only stride 1 allowed on non-range indexing.");g=g&&1===d.strides[k];let T=!!(d.beginMask&1<<k&&d.endMask&1<<k);if(d.beginValid&&d.endValid){if(N){let S=d.begin[k]<0?v+d.begin[k]:d.begin[k];if(d.begin[k]=S,d.end[k]=d.begin[k]+1,S<0||S>=v)throw Error(`slice index ${d.begin[k]} of dimension ${k} out of bounds.`)}else d.begin[k]=td(d.begin[k],0,d.strides[k],v,x,w),d.end[k]=td(d.end[k],1,d.strides[k],v,x,w);let I=1===d.strides[k]&&0===d.begin[k]&&d.end[k]===v;f=f&&I,m=m&&(0===k&&1===d.strides[k]||I)}else f=f&&1===d.strides[k]&&T,m=m&&(0===k&&1===d.strides[k]||T);let _,E=!1;if(d.beginValid&&d.endValid?(_=d.end[k]-d.begin[k],E=!0):N?(_=1,E=!0):T&&v>=0&&(_=d.strides[k]<0?-v:v,E=!0),E){let A;A=0===_||_<0!=d.strides[k]<0?0:Math.trunc(_/d.strides[k])+(_%d.strides[k]!=0?1:0),y.push(A)}else y.push(-1)}for(let M=0;M<d.finalShapeGatherIndices.length;++M){let D=d.finalShapeGatherIndices[M];D>=0?b.push(y[D]):-2===D&&b.push(1)}let $=b.filter((e,t)=>-2!==d.finalShapeGatherIndices[t]);return{finalShapeSparse:$,finalShape:b,isIdentity:f,sliceDim0:m,isSimpleSlice:g,begin:d.begin,end:d.end,strides:d.strides}}function td(e,t,n,r,a,s){if(a[t])return n>0?s[t]:s[t+1&1];{let i=e<0?r+e:e;return i<s[0]?s[0]:i>s[1]?s[1]:i}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tf{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}}class tm{constructor(){this.classNameMap={}}static getMap(){return null==tm.instance&&(tm.instance=new tm),tm.instance}static register(e){tm.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function tg(e){(0,S.hu)(null!=e.className,()=>"Class being registered does not have the static className property defined."),(0,S.hu)("string"==typeof e.className,()=>"className is required to be a string, but got type "+typeof e.className),(0,S.hu)(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),tm.register(e)}var ty=n(747),tb=n(3418);let tk=.1;function tN(e,t,n){return null==n&&(n=tv()),tx(e,t,(e,t)=>tI(e,t,n))}function tv(){return 32===m.BV.backend.floatPrecision()?.001:tk}function tx(e,t,n){let r=!0;if(((0,S.fU)(e)||(0,S.fU)(t))&&(r=!1),(0,S.fU)(e)&&(0,S.fU)(t)&&(r=!0),r){let a=e.constructor.name,s=t.constructor.name;if(a!==s)throw Error(`Arrays are of different type. Actual: ${a}. Expected: ${s}`)}if(Array.isArray(e)&&Array.isArray(t)){let i=(0,eG.C)(e),o=(0,eG.C)(t);if(!(0,S.cO)(i,o))throw Error(`Arrays have different shapes. Actual: [${i}]. Expected: [${o}]`)}let u=(0,S.fU)(e)?e:(0,S.xH)(e),l=(0,S.fU)(t)?t:(0,S.xH)(t);if(u.length!==l.length)throw Error(`Arrays have different lengths actual: ${u.length} vs expected: ${l.length}.
Actual:   ${u}.
Expected: ${l}.`);for(let p=0;p<l.length;++p){let c=u[p],h=l[p];if(!n(c,h))throw Error(`Arrays differ: actual[${p}] = ${c}, expected[${p}] = ${h}.
Actual:   ${u}.
Expected: ${l}.`)}"undefined"!=typeof expect&&expect().nothing()}function tw(e,t){e().then(()=>t.fail(),()=>t()),"undefined"!=typeof expect&&expect().nothing()}function tT(e,t){return(0,S.HD)(e)||(0,S.HD)(e[0])||(0,S.HD)(t)||(0,S.HD)(t[0])?tx(e,"string"==typeof t||"number"==typeof t||"boolean"==typeof t?[t]:t,(e,t)=>e==t):tx(e,t,(e,t)=>tI(e,t,0))}function tS(e,t,n){if(null==n&&(n=tv()),!tI(e,t,n))throw Error(`Numbers differ: actual === ${e}, expected === ${t}`);"undefined"!=typeof expect&&expect().nothing()}function tI(e,t,n){return!(isFinite(e)||isFinite(t))||!(isNaN(e)||isNaN(t)||Math.abs(e-t)>n)}function t_(e,t,n){for(let r=0;r<e.length;r++)if(e[r]<t||e[r]>n)throw Error(`Value out of range:${e[r]} low: ${t}, high: ${n}`)}function tE(e,t){let n=new Float32Array(e),r=new Float32Array(t);if(n.length!==r.length)throw Error(`Expected ArrayBuffer to be of length ${r.length}, but it was ${n.length}`);for(let a=0;a<r.length;a++)if(n[a]!==r[a])throw Error(`Expected ArrayBuffer value at ${a} to be ${r[a]} but got ${n[a]} instead`)}function tA(e){let t=document.createElement("video");return"playsInline"in t&&(t.playsInline=!0),t.muted=!0,t.loop=!0,t.style.position="fixed",t.style.left="0px",t.style.top="0px",t.preload="auto",t.appendChild(e),new Promise(e=>{t.addEventListener("loadeddata",n=>e(t)),t.load()})}async function tM(e){await e.play(),"requestVideoFrameCallback"in e&&await new Promise(t=>{e.requestVideoFrameCallback(t)})}/** @license See the LICENSE file. */ let tD="4.1.0";var t$=n(4368),tF=n(6407),tB=n(1274),tO=n(4841),tR=n(3261),tC=n(248),tV=n(6577),tP=n(633),tL=n(9494);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tz extends tf{minimize(e,t=!1,n){let{value:r,grads:a}=this.computeGradients(e,n);if(null!=n){let s=n.map(e=>({name:e.name,tensor:a[e.name]}));this.applyGradients(s)}else this.applyGradients(a);return((0,t$.B9)(a),t)?r:(r.dispose(),null)}get iterations(){return null==this.iterations_&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return(0,tP.pn)(e,t)}dispose(){null!=this.iterations_&&(0,t$.B9)(this.iterations_)}async saveIterations(){return null==this.iterations_&&(this.iterations_=0),{name:"iter",tensor:(0,tL.i)(this.iterations_,"int32")}}async getWeights(){throw Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(tz,Symbol.hasInstance,{value:e=>null!=e.minimize&&null!=e.computeGradients&&null!=e.applyGradients});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tW extends tz{constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],null==n&&(this.epsilon=m.BV.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);t.forEach((t,n)=>{let r=m.BV.registeredVariables[t];null==this.accumulatedGrads[n]&&(this.accumulatedGrads[n]={originalName:`${t}/accum_grad`,variable:(0,t$.lu)(()=>(0,tV.P)(r).variable(!1))}),null==this.accumulatedUpdates[n]&&(this.accumulatedUpdates[n]={originalName:`${t}/accum_var`,variable:(0,t$.lu)(()=>(0,tV.P)(r).variable(!1))});let a=Array.isArray(e)?e[n].tensor:e[t];if(null==a)return;let s=this.accumulatedGrads[n].variable,i=this.accumulatedUpdates[n].variable;(0,t$.lu)(()=>{let e=(0,tF.I)((0,tO.d)(s,this.rho),(0,tO.d)((0,tC.h)(a),1-this.rho)),t=(0,tO.d)((0,tB.h)((0,tR._)((0,tF.I)(i,this.epsilon)),(0,tR._)((0,tF.I)(s,this.epsilon))),a),n=(0,tF.I)((0,tO.d)(i,this.rho),(0,tO.d)((0,tC.h)(t),1-this.rho));s.assign(e),i.assign(n);let o=(0,tF.I)((0,tO.d)(t,-this.learningRate),r);r.assign(o)})}),this.incrementIterations()}dispose(){null!=this.accumulatedUpdates&&((0,t$.B9)(this.accumulatedGrads.map(e=>e.variable)),(0,t$.B9)(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){let e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=e.length/2;this.accumulatedGrads=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedUpdates=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}}tW.className="Adadelta",tg(tW);var tU=n(4006);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tG extends tz{constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);t.forEach((t,n)=>{let r=m.BV.registeredVariables[t];null==this.accumulatedGrads[n]&&(this.accumulatedGrads[n]={originalName:`${t}/accumulator`,variable:(0,t$.lu)(()=>(0,tU.h)(r.shape,this.initialAccumulatorValue).variable(!1))});let a=Array.isArray(e)?e[n].tensor:e[t];if(null==a)return;let s=this.accumulatedGrads[n].variable;(0,t$.lu)(()=>{let e=(0,tF.I)(s,(0,tC.h)(a));s.assign(e);let t=(0,tF.I)((0,tO.d)((0,tB.h)(a,(0,tR._)((0,tF.I)(e,m.BV.backend.epsilon()))),-this.learningRate),r);r.assign(t)})}),this.incrementIterations()}dispose(){null!=this.accumulatedGrads&&(0,t$.B9)(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulatedGrads=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}}tG.className="Adagrad",tg(tG);var tq=n(3453),tH=n(827);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tj extends tz{constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],(0,t$.lu)(()=>{this.accBeta1=(0,tL.i)(t).variable(),this.accBeta2=(0,tL.i)(n).variable()}),null==r&&(this.epsilon=m.BV.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);(0,t$.lu)(()=>{let n=(0,tH.l)(1,this.accBeta1),r=(0,tH.l)(1,this.accBeta2);t.forEach((t,a)=>{let s=m.BV.registeredVariables[t];null==this.accumulatedFirstMoment[a]&&(this.accumulatedFirstMoment[a]={originalName:`${t}/m`,variable:(0,t$.lu)(()=>(0,tV.P)(s).variable(!1))}),null==this.accumulatedSecondMoment[a]&&(this.accumulatedSecondMoment[a]={originalName:`${t}/v`,variable:(0,t$.lu)(()=>(0,tV.P)(s).variable(!1))});let i=Array.isArray(e)?e[a].tensor:e[t];if(null==i)return;let o=this.accumulatedFirstMoment[a].variable,u=this.accumulatedSecondMoment[a].variable,l=(0,tF.I)((0,tO.d)(o,this.beta1),(0,tO.d)(i,1-this.beta1)),p=(0,tF.I)((0,tO.d)(u,this.beta2),(0,tO.d)((0,tC.h)(i),1-this.beta2)),c=(0,tB.h)(l,n),h=(0,tB.h)(p,r);o.assign(l),u.assign(p);let d=(0,tF.I)((0,tO.d)((0,tB.h)(c,(0,tF.I)((0,tR._)(h),this.epsilon)),-this.learningRate),s);s.assign(d)}),this.accBeta1.assign((0,tO.d)(this.accBeta1,this.beta1)),this.accBeta2.assign((0,tO.d)(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),null!=this.accumulatedFirstMoment&&(0,t$.B9)(this.accumulatedFirstMoment.map(e=>e.variable)),null!=this.accumulatedSecondMoment&&(0,t$.B9)(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),(0,t$.lu)(()=>{this.accBeta1.assign((0,tq.s)(this.beta1,this.iterations_+1)),this.accBeta2.assign((0,tq.s)(this.beta2,this.iterations_+1))});let t=e.length/2;this.accumulatedFirstMoment=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedSecondMoment=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}}tj.className="Adam",tg(tj);var tK=n(6235),tX=n(632);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tZ extends tz{constructor(e,t,n,r=null,a=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=a,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],(0,t$.lu)(()=>{this.iteration=(0,tL.i)(0).variable(),this.accBeta1=(0,tL.i)(t).variable()}),null==r&&(this.epsilon=m.BV.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);(0,t$.lu)(()=>{let n=(0,tH.l)(1,this.accBeta1),r=(0,tB.h)(-this.learningRate,(0,tF.I)((0,tO.d)(this.iteration,this.decay),1));t.forEach((t,a)=>{let s=m.BV.registeredVariables[t];null==this.accumulatedFirstMoment[a]&&(this.accumulatedFirstMoment[a]={originalName:`${t}/m`,variable:(0,tV.P)(s).variable(!1)}),null==this.accumulatedWeightedInfNorm[a]&&(this.accumulatedWeightedInfNorm[a]={originalName:`${t}/v`,variable:(0,tV.P)(s).variable(!1)});let i=Array.isArray(e)?e[a].tensor:e[t];if(null==i)return;let o=this.accumulatedFirstMoment[a].variable,u=this.accumulatedWeightedInfNorm[a].variable,l=(0,tF.I)((0,tO.d)(o,this.beta1),(0,tO.d)(i,1-this.beta1)),p=(0,tO.d)(u,this.beta2),c=(0,tK.W)(i),h=(0,tX.g)(p,c);o.assign(l),u.assign(h);let d=(0,tF.I)((0,tO.d)((0,tB.h)(r,n),(0,tB.h)(l,(0,tF.I)(h,this.epsilon))),s);s.assign(d)}),this.iteration.assign((0,tF.I)(this.iteration,1)),this.accBeta1.assign((0,tO.d)(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),null!=this.accumulatedFirstMoment&&(0,t$.B9)(this.accumulatedFirstMoment.map(e=>e.variable)),null!=this.accumulatedWeightedInfNorm&&(0,t$.B9)(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}}tZ.className="Adamax",tg(tZ);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tQ extends tz{constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);t.forEach((t,n)=>{let r=Array.isArray(e)?e[n].tensor:e[t];if(null==r)return;let a=m.BV.registeredVariables[t];(0,t$.lu)(()=>{let e=(0,tF.I)((0,tO.d)(this.c,r),a);a.assign(e)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,null!=this.c&&this.c.dispose(),this.c=(0,t$.Cn)((0,tL.i)(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(0!==(e=await this.extractIterations(e)).length)throw Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}tQ.className="SGD",tg(tQ);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tY extends tQ{constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=(0,tL.i)(this.momentum)}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);t.forEach((t,n)=>{let r=m.BV.registeredVariables[t];null==this.accumulations[n]&&(this.accumulations[n]={originalName:`${t}/momentum`,variable:(0,t$.lu)(()=>(0,tV.P)(r).variable(!1))});let a=this.accumulations[n].variable,s=Array.isArray(e)?e[n].tensor:e[t];null!=s&&(0,t$.lu)(()=>{let e,t=(0,tF.I)((0,tO.d)(this.m,a),s);e=this.useNesterov?(0,tF.I)((0,tO.d)(this.c,(0,tF.I)(s,(0,tO.d)(t,this.m))),r):(0,tF.I)((0,tO.d)(this.c,t),r),a.assign(t),r.assign(e)})}),this.incrementIterations()}dispose(){this.m.dispose(),null!=this.accumulations&&(0,t$.B9)(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulations=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}}tY.className="Momentum",tg(tY);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tJ extends tz{constructor(e,t=.9,n=0,r=null,a=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=a,null==r&&(this.epsilon=m.BV.backend.epsilon()),null==e)throw Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);t.forEach((t,n)=>{let r=m.BV.registeredVariables[t];null==this.accumulatedMeanSquares[n]&&(this.accumulatedMeanSquares[n]={originalName:`${t}/rms`,variable:(0,t$.lu)(()=>(0,tV.P)(r).variable(!1))}),null==this.accumulatedMoments[n]&&(this.accumulatedMoments[n]={originalName:`${t}/momentum`,variable:(0,t$.lu)(()=>(0,tV.P)(r).variable(!1))}),null==this.accumulatedMeanGrads[n]&&this.centered&&(this.accumulatedMeanGrads[n]={originalName:`${t}/mg`,variable:(0,t$.lu)(()=>(0,tV.P)(r).variable(!1))});let a=Array.isArray(e)?e[n].tensor:e[t];if(null==a)return;let s=this.accumulatedMeanSquares[n].variable,i=this.accumulatedMoments[n].variable;(0,t$.lu)(()=>{let e=(0,tF.I)((0,tO.d)(s,this.decay),(0,tO.d)((0,tC.h)(a),1-this.decay));if(this.centered){let t=this.accumulatedMeanGrads[n].variable,o=(0,tF.I)((0,tO.d)(t,this.decay),(0,tO.d)(a,1-this.decay)),u=(0,tB.h)((0,tO.d)(a,this.learningRate),(0,tR._)((0,tH.l)(e,(0,tF.I)((0,tC.h)(o),this.epsilon)))),l=(0,tF.I)((0,tO.d)(i,this.momentum),u);s.assign(e),t.assign(o),i.assign(l);let p=(0,tH.l)(r,l);r.assign(p)}else{let c=(0,tF.I)((0,tO.d)(s,this.decay),(0,tO.d)((0,tC.h)(a),1-this.decay)),h=(0,tF.I)((0,tO.d)(i,this.momentum),(0,tB.h)((0,tO.d)(a,this.learningRate),(0,tR._)((0,tF.I)(c,this.epsilon))));s.assign(c),i.assign(h);let d=(0,tH.l)(r,h);r.assign(d)}})}),this.incrementIterations()}dispose(){null!=this.accumulatedMeanSquares&&(0,t$.B9)(this.accumulatedMeanSquares.map(e=>e.variable)),null!=this.accumulatedMeanGrads&&this.centered&&(0,t$.B9)(this.accumulatedMeanGrads.map(e=>e.variable)),null!=this.accumulatedMoments&&(0,t$.B9)(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=this.centered?e.length/3:e.length/2;this.accumulatedMeanSquares=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedMoments=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.centered&&(this.accumulatedMeanGrads=e.slice(2*t,3*t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}}tJ.className="RMSProp",tg(tJ);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class t0{static sgd(e){return new tQ(e)}static momentum(e,t,n=!1){return new tY(e,t,n)}static rmsprop(e,t=.9,n=0,r=null,a=!1){return new tJ(e,t,n,r,a)}static adam(e=.001,t=.9,n=.999,r=null){return new tj(e,t,n,r)}static adadelta(e=.001,t=.95,n=null){return new tW(e,t,n)}static adamax(e=.002,t=.9,n=.999,r=null,a=0){return new tZ(e,t,n,r,a)}static adagrad(e,t=.1){return new tG(e,t)}}var t1=n(1221),t2=n(2071),t3=n(9876);let t6={sgd:t0.sgd,momentum:t0.momentum,adadelta:t0.adadelta,adagrad:t0.adagrad,rmsprop:t0.rmsprop,adamax:t0.adamax,adam:t0.adam},t4="undefined"!=typeof requestAnimationFrame?requestAnimationFrame:"undefined"!=typeof setImmediate?setImmediate:e=>e();function t5(){return new Promise(e=>t4(()=>e()))}var t8=n(3591);/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function t7(e,t){let n=e[0].length;e.forEach((e,t)=>{S.hu(e.length===n,()=>`Error in concat${n}D: rank of tensors[${t}] must be the same as the rank of the rest (${n})`)}),S.hu(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);let r=e[0];e.forEach((e,a)=>{for(let s=0;s<n;s++)S.hu(s===t||e[s]===r[s],()=>`Error in concat${n}D: Shape of tensors[${a}] (${e}) does not match the shape of the rest (${r}) along the non-concatenated axis ${a}.`)})}function t9(e,t){let n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}var ne,nt=n(2582),nn=n(9323);function nr(e,t,n){let r=[];if(null==n&&null==t)return r;if(null==t)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(null==n)return r;if(e+n.length!==r.length)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let a=1;a<n.length;++a){let s=n[a],i=r[r.length-n.length+a],o=r[i];if(s>=0){if(o>=0){if(o!==s)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${a+e}] = ${s} but shape[${a+e}] = ${o}`)}else r[i]=s}}return r}function na(e){let t={FIRST_DIM_SIZE:r.FIRST_DIM_SIZE,VALUE_ROWIDS:r.VALUE_ROWIDS,ROW_LENGTHS:r.ROW_LENGTHS,ROW_SPLITS:r.ROW_SPLITS,ROW_LIMITS:r.ROW_LIMITS,ROW_STARTS:r.ROW_STARTS},n=[];for(let a of e)if(a in t)n.push(t[a]);else break;return n}function ns(e){return 0===e.length?0:e[0]===r.FIRST_DIM_SIZE?e.length-1:e.length}function ni(e,t){if(null==e||null==t)return;let n=e.length,r=t.length;if(n>=r)throw Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let a=0;a<Math.min(n,r-1);++a){let s=e[a],i=t[a+1];if(s>=0&&i>=0&&1!==s&&s!==i)throw Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${a-e.length}] = ${s} but ragged tensor input.flatValues.shape[${a-e.length}] = ${i}`)}}(ne=r||(r={}))[ne.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",ne[ne.VALUE_ROWIDS=1]="VALUE_ROWIDS",ne[ne.ROW_LENGTHS=2]="ROW_LENGTHS",ne[ne.ROW_SPLITS=3]="ROW_SPLITS",ne[ne.ROW_LIMITS=4]="ROW_LIMITS",ne[ne.ROW_STARTS=5]="ROW_STARTS";/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let no=30;function nu(e){return e<=no?e:(0,S.jP)(e,Math.floor(Math.sqrt(e)))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nl(e,t,n){let r=n*("number"==typeof e?e:e[0]),a=t*("number"==typeof e?e:e[1]);return[r,a]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function np(e,t,n,r=!0){let a=[];if(r)(a=a.concat(t.slice(0))).push(e[0]/n),a=a.concat(e.slice(1));else{a=a.concat(e[0]);let s=t.length;for(let i=0;i<s;++i)a=a.concat([e[i+1]/t[i],t[i]]);a=a.concat(e.slice(s+1))}return a}function nc(e,t,n=!0){let r=[];if(n){r.push(t);for(let a=t+1;a<e;++a)a<=2*t?(r.push(a),r.push(a-(t+1))):r.push(a)}else{let s=[],i=[];for(let o=1;o<e;++o)o>=2*t+1||o%2==1?i.push(o):s.push(o);r.push(...s),r.push(0),r.push(...i)}return r}function nh(e,t,n,r=!0){let a=[];r?a.push(e[0]/n):a.push(e[0]*n);for(let s=1;s<e.length;++s)s<=t.length?r?a.push(t[s-1]*e[s]):a.push(e[s]/t[s-1]):a.push(e[s]);return a}function nd(e,t){let n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function nf(e,t,n){let r=e.slice(0,1);for(let a=0;a<n;++a)r.push(e[a+1]-t[a][0]-t[a][1]);return r}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let nm=1.7580993408473768,ng=1.0507009873554805,ny=.3275911,nb=.254829592,nk=-.284496736,nN=1.421413741,nv=-1.453152027,nx=1.061405429;var nw=n(4706);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nT(e,t){if(e.length!==t.length)throw Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);let n=new Float32Array(2*e.length);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function nS(e){let t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function nI(e){let t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let a=0;a<e.length;a+=4)n[Math.floor(a/4)]=e[a],r[Math.floor(a/4)]=e[a+1];return{real:n,imag:r}}function n_(e){let t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let a=2;a<e.length;a+=4)n[Math.floor(a/4)]=e[a],r[Math.floor(a/4)]=e[a+1];return{real:n,imag:r}}function nE(e,t){let n=e[2*t],r=e[2*t+1];return{real:n,imag:r}}function nA(e,t,n,r){e[2*r]=t,e[2*r+1]=n}function nM(e,t){let n=new Float32Array(e/2),r=new Float32Array(e/2);for(let a=0;a<Math.ceil(e/2);a++){let s=(t?2:-2)*Math.PI*(a/e);n[a]=Math.cos(s),r[a]=Math.sin(s)}return{real:n,imag:r}}function nD(e,t,n){let r=(n?2:-2)*Math.PI*(e/t),a=Math.cos(r),s=Math.sin(r);return{real:a,imag:s}}let n$=/->/g;function nF(e,t){e=e.replace(/\s/g,"");let n=(e.length-e.replace(n$,"").length)/2;if(n<1)throw Error("Equations without an arrow are not supported.");if(n>1)throw Error('Equation must contain exactly one arrow ("->").');let[r,a]=e.split("->");(0,S.hu)(-1===r.indexOf("..."),()=>'The ellipsis notation ("...") is not supported yet.');let s=r.split(","),i=s.length;if(t!==i)throw Error(`Expected ${i} input tensors, received ${t}`);if(i>2)throw Error("Support for more than 2 input tensors is not implemented yet.");let o=[];for(let u=0;u<a.length;++u){let l=a[u];if(!s.some(e=>-1!==e.indexOf(l)))throw Error(`Output subscripts contain the label ${l} not present in the input subscripts.`);-1===o.indexOf(l)&&o.push(l)}for(let p=0;p<r.length;++p){let c=r[p];-1===o.indexOf(c)&&","!==c&&o.push(c)}let h=Array(s.length);for(let d=0;d<i;++d){if(new Set(s[d].split("")).size!==s[d].length)throw Error(`Found duplicate axes in input component ${s[d]}. Support for duplicate axes in input is not implemented yet.`);h[d]=[];for(let f=0;f<s[d].length;++f)h[d].push(o.indexOf(s[d][f]))}let m=o.length,g=a.length,y=[];for(let b=g;b<m;++b)y.push(b);return{allDims:o,summedDims:y,idDims:h}}function nB(e,t){let n=Array(e);n.fill(-1);for(let r=0;r<t.length;++r)n[t[r]]=r;let a=[];for(let s=0;s<e;++s)-1===n[s]&&a.push(s);return n=n.filter(e=>-1!==e),{permutationIndices:n,expandDims:a}}function nO(e,t,n){let r=Array(e);for(let a=0;a<n.length;++a){let s=n[a].shape;for(let i=0;i<t[a].length;++i)void 0===r[t[a][i]]?r[t[a][i]]=s[i]:(0,S.hu)(r[t[a][i]]===s[i],()=>`Expected dimension ${r[t[a][i]]} at axis ${i} of input shaped ${JSON.stringify(s)}, but got dimension ${s[i]}`)}}function nR(e,t){let n=[],r=0;0===e.length&&e.push(-1),r=e.length+1;for(let a=0;a<r;++a)n.push([]);let s=[];for(let i=0;i<e.length;++i){let o=e[i],u=nV(t,o);for(let l of u)-1===s.indexOf(l)&&(n[i].push(l),s.push(l))}return{path:e,steps:n}}function nC(e){return e.every((e,t)=>e===t)}function nV(e,t){let n=[];for(let r=0;r<e.length;++r)(0===e[r].length||-1!==e[r].indexOf(t)||-1===t)&&n.push(r);return n}function nP(e,t,n=0){let r=[];if("number"==typeof t)(0,S.hu)(e.shape[n]%t==0,()=>"Number of splits must evenly divide the axis."),r=Array(t).fill(e.shape[n]/t);else{let a=t.reduce((e,t)=>(-1===t&&(e+=1),e),0);(0,S.hu)(a<=1,()=>"There should be only one negative value in split array.");let s=t.indexOf(-1);if(-1!==s){let i=t.reduce((e,t)=>t>0?e+t:e);t[s]=e.shape[n]-i}(0,S.hu)(e.shape[n]===t.reduce((e,t)=>e+t),()=>"The sum of sizes must match the size of the axis dimension."),r=t}return r}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nL(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function nz(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function nW(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nU(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function nG(e,t){return`size ${e} must be non-negative, not ${t}`}function nq(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function nH(e,t){let n=(0,S.NA)(e),r=(0,S.NA)(t);return`Input to reshape is a SparseTensor with ${n}
  dense values, but the requested shape requires a multiple of ${r}. inputShape=${e} outputShape= ${t}`}function nj(e,t){let n=(0,S.NA)(e),r=(0,S.NA)(t);return`Input to reshape is a tensor with ${n} dense values, but the requested shape has ${r}. inputShape=${e} outputShape=${t}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nK(){return"segment ids must be >= 0"}function nX(){return"segment ids are not increasing"}function nZ(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function nQ(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nY(e,t){let n=!1,r;for(e<=no?(r=e,n=!0):r=(0,S.jP)(e,Math.floor(Math.sqrt(e)));!n;)r>t||r===e?n=!0:r=(0,S.jP)(e,r+1);return r}function nJ(e,t,n){let r=[],a=e.length;for(let s=0;s<a;s++)s!==t?r.push(e[s]):r.push(n);return r}function n0(e,t,n,r){let a=t.shape.length,s=e.shape.length;if(0!==r&&(r<-a||r>a))throw Error(`Expect batchDims in the range of [-${a}, ${a}], but got ${r}`);if(r<0&&(r+=a),r>s)throw Error(`batchDims (${r}) must be less than rank(x) (
    ${s}).`);if(n<r)throw Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let i=0;i<r;++i)if(e.shape[i]!==t.shape[i])throw Error(`x.shape[${i}]: ${e.shape[i]} should be equal to indices.shape[${i}]: ${t.shape[i]}.`);let o=e.shape[n],u=[],l=1,p=1,c=1;for(let h=0;h<r;++h)u.push(e.shape[h]),l*=e.shape[h];for(let d=r;d<n;d++)u.push(e.shape[d]),p*=e.shape[d];for(let f=r;f<a;f++)u.push(t.shape[f]);for(let m=n+1;m<s;m++)u.push(e.shape[m]),c*=e.shape[m];return{batchSize:l,sliceSize:c,outerSize:p,dimSize:o,outputShape:u}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function n1(e){try{return e.map(e=>(0,tb.decodeString)(e))}catch(t){throw Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function n2(e){return e.map(e=>(0,tb.encodeString)(e))}var n3=n(8329),n6=n(8333),n4=n(8713);/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ },9121:function(e,t,n){"use strict";n.d(t,{$HU:function(){return ts},$g6:function(){return J},$w:function(){return K},Acj:function(){return eg},BMI:function(){return eV},BiW:function(){return tn},Byc:function(){return L},CAk:function(){return eS},CQl:function(){return tt},D2d:function(){return tO},DlI:function(){return e9},Eh3:function(){return D},FKq:function(){return tI},G3Y:function(){return tK},GBy:function(){return t_},Gcp:function(){return tM},HEU:function(){return et},HZH:function(){return to},Hhh:function(){return eR},Hmb:function(){return tl},IKK:function(){return l},IMb:function(){return N},J$2:function(){return ew},J_u:function(){return ek},JhU:function(){return y},Kgp:function(){return e5},L8s:function(){return tA},Ly9:function(){return T},M2y:function(){return h},MIZ:function(){return tG},MRv:function(){return tS},MZg:function(){return eD},NEP:function(){return ea},NZg:function(){return eX},O3z:function(){return tD},OAf:function(){return ez},OR:function(){return ep},OU7:function(){return eW},OV7:function(){return eL},Omj:function(){return en},Oyi:function(){return m},PYm:function(){return eA},PhF:function(){return tb},QCc:function(){return g},QRR:function(){return U},Qg5:function(){return eb},QiL:function(){return e6},Qvg:function(){return tQ},RFZ:function(){return I},ROF:function(){return b},RQH:function(){return tv},RuY:function(){return tJ},SX0:function(){return ee},SYM:function(){return r},SbG:function(){return th},SpW:function(){return s},T0n:function(){return G},TQc:function(){return tE},TR1:function(){return P},ToN:function(){return tZ},Tr8:function(){return tW},Uyb:function(){return el},VGw:function(){return a},Vbg:function(){return eK},VcC:function(){return W},VfG:function(){return eM},Vn9:function(){return Q},W0H:function(){return e1},XDQ:function(){return tY},XLW:function(){return v},XkS:function(){return tz},Xze:function(){return o},Y0y:function(){return ei},YFo:function(){return es},YoZ:function(){return eC},ZbH:function(){return e_},ZjV:function(){return tB},Zz9:function(){return A},_JP:function(){return tP},_V0:function(){return t6},_Yw:function(){return tp},_k9:function(){return k},_tC:function(){return tR},a5O:function(){return tT},aJk:function(){return c},avt:function(){return eN},b9H:function(){return t2},bK0:function(){return tC},bV0:function(){return tm},c17:function(){return eq},cWu:function(){return tH},cie:function(){return q},cye:function(){return e0},dDz:function(){return te},deh:function(){return eu},dpD:function(){return tu},e07:function(){return tf},e6w:function(){return tr},e7N:function(){return eI},eBW:function(){return t1},eEB:function(){return S},eZ0:function(){return eO},ekb:function(){return Z},gJX:function(){return _},h8e:function(){return t0},hdR:function(){return er},i5y:function(){return tw},iHb:function(){return z},iJz:function(){return ey},iWB:function(){return ev},iZT:function(){return em},ik2:function(){return C},jMg:function(){return f},jQk:function(){return tV},jQs:function(){return ej},jeX:function(){return ec},kU:function(){return eE},kpP:function(){return tX},kuV:function(){return eQ},luS:function(){return t4},lyA:function(){return e4},mKl:function(){return td},mTV:function(){return eP},mc4:function(){return V},mhS:function(){return $},mm_:function(){return i},n9L:function(){return tq},nhH:function(){return t$},nr8:function(){return ty},o0g:function(){return e7},o2y:function(){return R},oFR:function(){return tk},oHH:function(){return Y},oT6:function(){return u},p2w:function(){return tN},p4S:function(){return X},pe_:function(){return e8},q1x:function(){return ef},q2K:function(){return eG},q8u:function(){return eH},qCd:function(){return eF},qIC:function(){return eB},qWM:function(){return e2},qi_:function(){return ed},qkr:function(){return ti},qw7:function(){return d},r7n:function(){return ex},s1s:function(){return tL},sEM:function(){return tU},sHE:function(){return eh},sJF:function(){return p},sL$:function(){return H},usg:function(){return t3},uv1:function(){return eJ},vFR:function(){return eU},vtC:function(){return eT},vwp:function(){return eo},w3H:function(){return tF},w6g:function(){return e$},wUP:function(){return F},wYB:function(){return tx},wYn:function(){return eZ},we_:function(){return e3},wm:function(){return B},wx7:function(){return tj},x12:function(){return O},xJR:function(){return ta},xQA:function(){return tg},xnO:function(){return E},y7R:function(){return j},yQU:function(){return eY},yj2:function(){return M},zbQ:function(){return tc},zvY:function(){return w},zws:function(){return x}});let r="Abs",a="Acos",s="Acosh",i="Add",o="AddN",u="All",l="Any",p="ArgMax",c="ArgMin",h="Asin",d="Asinh",f="Atan",m="Atanh",g="Atan2",y="AvgPool",b="AvgPoolGrad",k="AvgPool3D",N="AvgPool3DGrad",v="BatchMatMul",x="BatchToSpaceND",w="Bincount",T="BroadcastTo",S="BroadcastArgs",I="Cast",_="Ceil",E="ClipByValue",A="Complex",M="ComplexAbs",D="Concat",$="Conv2D",F="Conv2DBackpropFilter",B="Conv2DBackpropInput",O="Conv3D",R="Conv3DBackpropFilterV2",C="Conv3DBackpropInputV2",V="Cos",P="Cosh",L="Cumprod",z="Cumsum",W="CropAndResize",U="DenseBincount",G="DepthToSpace",q="DepthwiseConv2dNative",H="DepthwiseConv2dNativeBackpropFilter",j="DepthwiseConv2dNativeBackpropInput",K="Diag",X="Dilation2D",Z="Dilation2DBackpropInput",Q="Dilation2DBackpropFilter",Y="RealDiv",J="Einsum",ee="Elu",et="EluGrad",en="Erf",er="Equal",ea="Exp",es="ExpandDims",ei="Expm1",eo="FFT",eu="Fill",el="FlipLeftRight",ep="Floor",ec="FloorDiv",eh="FusedBatchNorm",ed="GatherV2",ef="GatherNd",em="Greater",eg="GreaterEqual",ey="Identity",eb="IFFT",ek="Imag",eN="IsFinite",ev="IsInf",ex="IsNan",ew="LeakyRelu",eT="Less",eS="LessEqual",eI="LinSpace",e_="Log",eE="Log1p",eA="LogicalAnd",eM="LogicalNot",eD="LogicalOr",e$="LogicalXor",eF="LogSoftmax",eB="LowerBound",eO="LRN",eR="LRNGrad",eC="Max",eV="Maximum",eP="MaxPool",eL="MaxPoolGrad",ez="MaxPool3D",eW="MaxPool3DGrad",eU="MaxPoolWithArgmax",eG="Mean",eq="Min",eH="Minimum",ej="MirrorPad",eK="Mod",eX="Multinomial",eZ="Multiply",eQ="Neg",eY="NotEqual",eJ="NonMaxSuppressionV3",e0="NonMaxSuppressionV4",e1="NonMaxSuppressionV5",e2="OnesLike",e3="OneHot",e6="Pack",e4="PadV2",e5="Pool",e8="Pow",e7="Prelu",e9="Prod",te="RaggedGather",tt="RaggedRange",tn="RaggedTensorToTensor",tr="Range",ta="Real",ts="Reciprocal",ti="Relu",to="Reshape",tu="ResizeNearestNeighbor",tl="ResizeNearestNeighborGrad",tp="ResizeBilinear",tc="ResizeBilinearGrad",th="Relu6",td="Reverse",tf="Round",tm="Rsqrt",tg="ScatterNd",ty="SearchSorted",tb="Select",tk="Selu",tN="Slice",tv="Sin",tx="Sinh",tw="Sign",tT="Sigmoid",tS="Softplus",tI="Sqrt",t_="Sum",tE="SpaceToBatchND",tA="SplitV",tM="Softmax",tD="SparseFillEmptyRows",t$="SparseReshape",tF="SparseSegmentMean",tB="SparseSegmentSum",tO="SparseToDense",tR="SquaredDifference",tC="Square",tV="StridedSlice",tP="StringNGrams",tL="StringSplit",tz="StringToHashBucketFast",tW="Sub",tU="Tan",tG="Tanh",tq="Tile",tH="TopK",tj="Transform",tK="Transpose",tX="Unique",tZ="Unpack",tQ="UnsortedSegmentSum",tY="UpperBound",tJ="ZerosLike",t0="Step",t1="FromPixels",t2="RotateWithOffset",t3="_FusedMatMul",t6="FusedConv2D",t4="FusedDepthwiseConv2D"},6151:function(e,t,n){"use strict";n.d(t,{Li:function(){return h},T3:function(){return m},bt:function(){return f},nE:function(){return d},pI:function(){return u},tr:function(){return p},uk:function(){return l},wC:function(){return c}});var r=n(2885),a=n(5938),s=n(4706);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let i=(0,a.R)("kernelRegistry",()=>new Map),o=(0,a.R)("gradRegistry",()=>new Map);function u(e,t){let n=g(e,t);return i.get(n)}function l(e){return o.get(e)}function p(e){let t=i.entries(),n=[];for(;;){let{done:r,value:a}=t.next();if(r)break;let[s,o]=a,[u,]=s.split("_");u===e&&n.push(o)}return n}function c(e){let{kernelName:t,backendName:n}=e,r=g(t,n);i.has(r)&&s.Z(`The kernel '${t}' for backend '${n}' is already registered`),i.set(r,e)}function h(e){let{kernelName:t}=e;o.has(t)&&(0,r.OB)().getBool("DEBUG")&&s.Z(`Overriding the gradient for '${t}'`),o.set(t,e)}function d(e,t){let n=g(e,t);if(!i.has(n))throw Error(`The kernel '${e}' for backend '${t}' is not registered`);i.delete(n)}function f(e){if(!o.has(e))throw Error(`The gradient '${e}' for backend is not registered`);o.delete(e)}function m(e,t){let n=p(e);n.forEach(e=>{let n=Object.assign({},e,{backendName:t});c(n)})}function g(e,t){return`${t}_${e}`}},4706:function(e,t,n){"use strict";n.d(t,{Z:function(){return a},c:function(){return s}});var r=n(2885);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function a(...e){(0,r.OB)().getBool("IS_TEST")||(0,r.OB)().getBool("PROD")||console.warn(...e)}function s(...e){(0,r.OB)().getBool("IS_TEST")||(0,r.OB)().getBool("PROD")||console.log(...e)}},6235:function(e,t,n){"use strict";n.d(t,{W:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({abs_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","abs");return"complex64"===t.dtype?r.BV.runKernel(a.yj2,{x:t}):r.BV.runKernel(a.SYM,{x:t})}})},6407:function(e,t,n){"use strict";n.d(t,{I:function(){return u}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(2668);let u=(0,o.op)({add_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,i._1)(e,"a","add"),o=(0,i._1)(t,"b","add");[n,o]=(0,s.makeTypesMatch)(n,o);let u={a:n,b:o};return r.BV.runKernel(a.mm_,u)}})},3591:function(e,t,n){"use strict";n.d(t,{LJ:function(){return p},Q3:function(){return l},Vh:function(){return s},YB:function(){return a},kz:function(){return i},lB:function(){return u},rv:function(){return o},sY:function(){return c}});var r=n(569);/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function a(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function s(e,t,n){let r=e.length+t.length,a=[],s=0,i=0;for(let o=0;o<r;o++)-1===n.indexOf(o)?a.push(e[s++]):a.push(t[i++]);return a}function i(e,t){let n=[],r=e.length;for(let a=0;a<r;a++)-1===t.indexOf(a)&&n.push(e[a]);let s=t.map(t=>e[t]);return[n,s]}function o(e,t){let n=t.map(e=>1);return s(e,n,t)}function u(e,t,n){r.hu(a(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function l(e,t){if(a(e,t))return null;let n=[];for(let r=0;r<t;++r)-1===e.indexOf(r)&&n.push(r);return e.forEach(e=>n.push(e)),n}function p(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function c(e,t){let n=[];for(let r=t-e;r<t;++r)n.push(r);return n}},2200:function(e,t,n){"use strict";/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function r(e,t){let n=e.length,r=[];for(let a=0;a<n;a++){let s=n-1-a,i=e[s]||1,o=t[t.length-1-a]||1;o>1&&1===i&&r.unshift(s)}return r}function a(e,t){let n=[];for(let r=0;r<t.length;r++){let a=e[e.length-r-1],s=t.length-r-1,i=t[s];(null==a||1===a&&i>1)&&n.unshift(s)}return n}function s(e,t){let n=[],r=Math.max(e.length,t.length);for(let a=0;a<r;a++){let s=e[e.length-a-1];null==s&&(s=1);let i=t[t.length-a-1];if(null==i&&(i=1),1===s)n.unshift(i);else if(1===i)n.unshift(s);else if(s!==i){let o=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(o)}else n.unshift(s)}return n}n.r(t),n.d(t,{assertAndGetBroadcastShape:function(){return s},getBroadcastDims:function(){return r},getReductionAxes:function(){return a}})},2657:function(e,t,n){"use strict";n.d(t,{f:function(){return s}});var r=n(974),a=n(569);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function s(e,t="float32",n){return t=t||"float32",a.Mu(e),new r.YD(e,t,n)}},2271:function(e,t,n){"use strict";n.d(t,{p:function(){return u}});var r=n(196),a=n(9121),s=n(3740),i=n(569),o=n(2668);let u=(0,o.op)({cast_:/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,s._1)(e,"x","cast");if(!i.LP(t))throw Error(`Failed to cast to unknown dtype ${t}`);if("string"===t&&"string"!==n.dtype||"string"!==t&&"string"===n.dtype)throw Error("Only strings can be casted to strings");return r.BV.runKernel(a.RFZ,{x:n},{dtype:t})}})},8723:function(e,t,n){"use strict";n.d(t,{d:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({clone_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","clone","string_or_numeric");return r.BV.runKernel(a.iJz,{x:t})}})},1661:function(e,t,n){"use strict";n.d(t,{P:function(){return u}});var r=n(196),a=n(9121),s=n(3740),i=n(569),o=n(2668);let u=(0,o.op)({complex_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,s._1)(e,"real","complex"),o=(0,s._1)(t,"imag","complex");return i.k5(n.shape,o.shape,`real and imag shapes, ${n.shape} and ${o.shape}, must match in call to tf.complex().`),r.BV.runKernel(a.Zz9,{real:n,imag:o})}})},2582:function(e,t,n){"use strict";n.d(t,{I0:function(){return f},Ix:function(){return o},Rf:function(){return a},Xw:function(){return s},aO:function(){return l},jT:function(){return m},jw:function(){return u},m:function(){return y},pl:function(){return i},sl:function(){return g}});var r=n(569);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function a(e,t,n,r,a="NHWC",s){let i=e[3],u=g(a);return o(e,[...t,i],n,s,r,null,null,u)}function s(e,t,n,r,a,s,i="channelsLast"){let[u,l]=p(t),c;if("channelsLast"===i)c=[u,l,e[3],e[3]];else if("channelsFirst"===i)c=[u,l,e[1],e[1]];else throw Error(`Unknown dataFormat ${i}`);return o(e,c,n,r,a,s,!1,i)}function i(e,t,n,r,a,s,i="NDHWC"){let[o,l,p]=c(t),h,d;if("NDHWC"===i)d="channelsLast",h=[o,l,p,e[4],e[4]];else if("NCDHW"===i)d="channelsFirst",h=[o,l,p,e[1],e[1]];else throw Error(`Unknown dataFormat ${i}`);return u(e,h,n,r,a,!1,d,s)}function o(e,t,n,r,a,s,i=!1,o="channelsLast"){let[u,c,f,m]=[-1,-1,-1,-1];if("channelsLast"===o)[u,c,f,m]=e;else if("channelsFirst"===o)[u,m,c,f]=e;else throw Error(`Unknown dataFormat ${o}`);let[g,y,,b]=t,[k,N]=p(n),[v,x]=p(r),w=h(g,v),T=h(y,x),{padInfo:S,outHeight:I,outWidth:_}=function(e,t,n,r,a,s,i,o,u){let p,c,h;if("number"==typeof e){p={top:e,bottom:e,left:e,right:e,type:0===e?"VALID":"NUMBER"};let f=function(e,t,n,r,a){null==r&&(r=l(e,t,n));let s=e[0],i=e[1],o=d((s-t+2*r)/n+1,a),u=d((i-t+2*r)/n+1,a);return[o,u]}([t,n],s,r,e,o);c=f[0],h=f[1]}else if("same"===e){c=Math.ceil(t/r),h=Math.ceil(n/a);let m=Math.max(0,(c-1)*r+s-t),g=Math.max(0,(h-1)*a+i-n),y=Math.floor(m/2),b=Math.floor(g/2);p={top:y,bottom:m-y,left:b,right:g-b,type:"SAME"}}else if("valid"===e)p={top:0,bottom:0,left:0,right:0,type:"VALID"},c=Math.ceil((t-s+1)/r),h=Math.ceil((n-i+1)/a);else if("object"==typeof e){let k="channelsLast"===u?e[1][0]:e[2][0],N="channelsLast"===u?e[1][1]:e[2][1],v="channelsLast"===u?e[2][0]:e[3][0],x="channelsLast"===u?e[2][1]:e[3][1];p={top:k,bottom:N,left:v,right:x,type:0===k&&0===N&&0===v&&0===x?"VALID":"EXPLICIT"},c=d((t-s+k+N)/r+1,o),h=d((n-i+v+x)/a+1,o)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:p,outHeight:c,outWidth:h}}(a,c,f,k,N,w,T,s,o),E=i?b*m:b,A;return"channelsFirst"===o?A=[u,E,I,_]:"channelsLast"===o&&(A=[u,I,_,E]),{batchSize:u,dataFormat:o,inHeight:c,inWidth:f,inChannels:m,outHeight:I,outWidth:_,outChannels:E,padInfo:S,strideHeight:k,strideWidth:N,filterHeight:g,filterWidth:y,effectiveFilterHeight:w,effectiveFilterWidth:T,dilationHeight:v,dilationWidth:x,inShape:e,outShape:A,filterShape:t}}function u(e,t,n,r,a,s=!1,i="channelsLast",o){let[u,p,f,m,g]=[-1,-1,-1,-1,-1];if("channelsLast"===i)[u,p,f,m,g]=e;else if("channelsFirst"===i)[u,g,p,f,m]=e;else throw Error(`Unknown dataFormat ${i}`);let[y,b,k,,N]=t,[v,x,w]=c(n),[T,S,I]=c(r),_=h(y,T),E=h(b,S),A=h(k,I),{padInfo:M,outDepth:D,outHeight:$,outWidth:F}=function(e,t,n,r,a,s,i,o,u,p,c){let h,f,m,g;if("number"==typeof e){h={top:e,bottom:e,left:e,right:e,front:e,back:e,type:0===e?"VALID":"NUMBER"};let y=function(e,t,n,r,a,s){null==a&&(a=l(e,t,r));let i=e[0],o=e[1],u=e[2],p=d((i-t+2*a)/r+1,s),c=d((o-t+2*a)/r+1,s),h=d((u-t+2*a)/r+1,s);return[p,c,h,1]}([t,n,r,1],o,1,a,e,c);f=y[0],m=y[1],g=y[2]}else if("same"===e){f=Math.ceil(t/a),m=Math.ceil(n/s),g=Math.ceil(r/i);let b=(f-1)*a+o-t,k=(m-1)*s+u-n,N=(g-1)*i+p-r,v=Math.floor(b/2),x=Math.floor(k/2),w=Math.floor(N/2);h={top:x,bottom:k-x,left:w,right:N-w,front:v,back:b-v,type:"SAME"}}else if("valid"===e)h={top:0,bottom:0,left:0,right:0,front:0,back:0,type:"VALID"},f=Math.ceil((t-o+1)/a),m=Math.ceil((n-u+1)/s),g=Math.ceil((r-p+1)/i);else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:h,outDepth:f,outHeight:m,outWidth:g}}(a,p,f,m,v,x,w,_,E,A,o),B=s?N*g:N,O;return"channelsFirst"===i?O=[u,B,D,$,F]:"channelsLast"===i&&(O=[u,D,$,F,B]),{batchSize:u,dataFormat:i,inDepth:p,inHeight:f,inWidth:m,inChannels:g,outDepth:D,outHeight:$,outWidth:F,outChannels:B,padInfo:M,strideDepth:v,strideHeight:x,strideWidth:w,filterDepth:y,filterHeight:b,filterWidth:k,effectiveFilterDepth:_,effectiveFilterHeight:E,effectiveFilterWidth:A,dilationDepth:T,dilationHeight:S,dilationWidth:I,inShape:e,outShape:O,filterShape:t}}function l(e,t,n,r=1){let a=h(t,r);return Math.floor((e[0]*(n-1)-n+a)/2)}function p(e){return"number"==typeof e?[e,e,e]:2===e.length?[e[0],e[1],1]:e}function c(e){return"number"==typeof e?[e,e,e]:e}function h(e,t){return t<=1?e:e+(e-1)*(t-1)}function d(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw Error(`Unknown roundingMode ${t}`)}}function f(e){let[t,n,r]=p(e);return 1===t&&1===n&&1===r}function m(e,t){return f(e)||f(t)}function g(e){if("NHWC"===e)return"channelsLast";if("NCHW"===e)return"channelsFirst";throw Error(`Unknown dataFormat ${e}`)}function y(e,t,n){if(null!=n){if("string"==typeof t)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if("number"==typeof t)r.hu(r.GN(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if("object"==typeof t)t.forEach(t=>{t.forEach(t=>{r.hu(r.GN(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}},1274:function(e,t,n){"use strict";n.d(t,{h:function(){return l}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(9165),u=n(2668);let l=(0,u.op)({div_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,i._1)(e,"a","div"),u=(0,i._1)(t,"b","div");if([n,u]=(0,s.makeTypesMatch)(n,u),"int32"===n.dtype&&"int32"===u.dtype)return(0,o.q)(n,u);let l={a:n,b:u};return r.BV.runKernel(a.oHH,l,{})}})},3233:function(e,t,n){"use strict";n.d(t,{p:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({elu_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","elu","float32");return r.BV.runKernel(a.SX0,{x:t})}})},4006:function(e,t,n){"use strict";n.d(t,{h:function(){return i}});var r=n(196),a=n(9121),s=n(569);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function i(e,t,n){return(0,s.Mu)(e),r.BV.runKernel(a.deh,{},{shape:e,value:t,dtype:n})}},9165:function(e,t,n){"use strict";n.d(t,{q:function(){return u}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(2668);let u=(0,o.op)({floorDiv_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,i._1)(e,"a","floorDiv"),o=(0,i._1)(t,"b","floorDiv");[n,o]=(0,s.makeTypesMatch)(n,o);let u={a:n,b:o};return r.BV.runKernel(a.jeX,u)}})},9323:function(e,t,n){"use strict";n.d(t,{Fr:function(){return f},QH:function(){return g},pf:function(){return m},uy:function(){return y}});var r=n(2200),a=n(3233),s=n(9133),i=n(4841),o=n(8151),u=n(7409),l=n(3582),p=n(4968),c=n(625),h=n(1901),d=n(5475);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function f(e,t,n){if(null==n||"linear"===n)return e;if("relu"===n)return(0,i.d)(e,(0,h.N)(t));throw Error(`Cannot compute gradient for fused activation ${n}.`)}function m(e,t){let n=t,a=r.getReductionAxes(e.shape,t.shape);return a.length>0&&(n=(0,d.S)(n,a)),(0,p.X)(n,e.shape)}function g(e,t,n,r){if("linear"===t)return e;if("relu"===t)return(0,u.U)(e);if("elu"===t)return(0,a.p)(e);if("relu6"===t)return(0,l.b)(e);if("prelu"===t)return(0,o.A)(e,n);else if("leakyrelu"===t)return(0,s.h)(e,r);else if("sigmoid"===t)return(0,c.X)(e);throw Error(`Unknown fused activation ${t}.`)}let y=(e,t)=>!(e>0)||"linear"===t},4386:function(e,t,n){"use strict";n.d(t,{a:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({imag_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"input","imag");return r.BV.runKernel(a.J_u,{input:t})}})},9133:function(e,t,n){"use strict";n.d(t,{h:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({leakyRelu_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=.2){let n=(0,s._1)(e,"x","leakyRelu");return r.BV.runKernel(a.J$2,{x:n},{alpha:t})}})},9876:function(e,t,n){"use strict";var r,a;n.d(t,{I:function(){return r}}),(a=r||(r={}))[a.NONE=0]="NONE",a[a.MEAN=1]="MEAN",a[a.SUM=2]="SUM",a[a.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"},8687:function(e,t,n){"use strict";n.d(t,{O:function(){return u}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(2668);let u=(0,o.op)({matMul_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=!1,o=!1){let u=(0,i._1)(e,"a","matMul"),l=(0,i._1)(t,"b","matMul");[u,l]=(0,s.makeTypesMatch)(u,l);let p={a:u,b:l};return r.BV.runKernel(a.XLW,p,{transposeA:n,transposeB:o})}})},632:function(e,t,n){"use strict";n.d(t,{g:function(){return p}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(2200),u=n(2271),l=n(2668);let p=(0,l.op)({maximum_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,i._1)(e,"a","maximum"),l=(0,i._1)(t,"b","maximum");[n,l]=(0,s.makeTypesMatch)(n,l),"bool"===n.dtype&&(n=(0,u.p)(n,"int32"),l=(0,u.p)(l,"int32")),(0,o.assertAndGetBroadcastShape)(n.shape,l.shape);let p={a:n,b:l};return r.BV.runKernel(a.BMI,p)}})},4841:function(e,t,n){"use strict";n.d(t,{d:function(){return u}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(2668);let u=(0,o.op)({mul_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,i._1)(e,"a","mul"),o=(0,i._1)(t,"b","mul");[n,o]=(0,s.makeTypesMatch)(n,o);let u={a:n,b:o};return r.BV.runKernel(a.wYn,u)}})},7370:function(e,t,n){"use strict";n.d(t,{W:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({neg_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","neg");return r.BV.runKernel(a.kuV,{x:t})}})},6708:function(e,t,n){"use strict";n.d(t,{l:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({oneHot_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=1,i=0,o="int32"){if(t<2)throw Error(`Error in oneHot: depth must be >=2, but it is ${t}`);let u=(0,s._1)(e,"indices","oneHot","int32");return r.BV.runKernel(a.we_,{indices:u},{dtype:o,depth:t,onValue:n,offValue:i})}})},2668:function(e,t,n){"use strict";n.d(t,{op:function(){return i},z:function(){return s}});var r=n(196),a=n(569);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let s="__op";function i(e){let t=Object.keys(e);if(1!==t.length)throw Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0],i=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n+=s;let o=(...e)=>{r.BV.startScope(n);try{let t=i(...e);return(0,a.tI)(t)&&console.error("Cannot return a Promise inside of tidy."),r.BV.endScope(t),t}catch(s){throw r.BV.endScope(null),s}};return Object.defineProperty(o,"name",{value:n,configurable:!0}),o}},2071:function(e,t,n){"use strict";n.d(t,{zvA:function(){return u.z},WnP:function(){return a.W},Khb:function(){return l},__u:function(){return p},IHx:function(){return c.I},QBD:function(){return d},$6P:function(){return f},YjB:function(){return m},NqF:function(){return g},vHJ:function(){return y},ZRM:function(){return b},VfV:function(){return k},z4N:function(){return N},fvJ:function(){return x},C80:function(){return w},wS1:function(){return _},uR5:function(){return E},zEQ:function(){return R},tgs:function(){return V},Dxk:function(){return P},JY5:function(){return L},p3b:function(){return z},E4h:function(){return C},yE8:function(){return W},anm:function(){return nL},XsQ:function(){return U},UFq:function(){return G},f3b:function(){return q.f},pju:function(){return T.p},mDi:function(){return H},iUl:function(){return K},d9v:function(){return A.d},PYB:function(){return X.P},zoF:function(){return M},gME:function(){return Z},Izb:function(){return Y},MNy:function(){return ee},ZaL:function(){return en},PAt:function(){return ea},Tek:function(){return er},bc:function(){return ei},pdZ:function(){return eo},$QV:function(){return el},mCk:function(){return ep},f9Y:function(){return ec},mew:function(){return nX},$Gn:function(){return eh},zbp:function(){return ed},ppE:function(){return ef},nTT:function(){return em},B10:function(){return eg},Ka3:function(){return ey},WmZ:function(){return eb},hiC:function(){return ek.h},NTj:function(){return eT},AKD:function(){return eS},rvX:function(){return nj},WYO:function(){return eI},pyx:function(){return e_.p},GRh:function(){return nK},DgJ:function(){return ev},qNN:function(){return eE},d2q:function(){return eV},Qqt:function(){return eP},dt4:function(){return eL},t$B:function(){return ez},iyy:function(){return eU},kp_:function(){return np},hlL:function(){return j.h},GWj:function(){return eG},qPi:function(){return eq.q},imm:function(){return r},Iqj:function(){return eH},dbB:function(){return nH},pjt:function(){return ej},brS:function(){return eK},Sxn:function(){return nc},asL:function(){return eX.a},BHj:function(){return rP},V3u:function(){return nQ},wx0:function(){return nh},xVT:function(){return eZ},UWc:function(){return eQ},i2d:function(){return eY},hi7:function(){return eJ.h},d9m:function(){return e0},zN1:function(){return e1},$r2:function(){return rL},SX3:function(){return e2},G9k:function(){return e3},cM7:function(){return e6},Krr:function(){return e4},e_t:function(){return e9},CmS:function(){return tt},l_t:function(){return tn},HvI:function(){return tr},hJK:function(){return ta},K5V:function(){return ts},egP:function(){return ti},MB5:function(){return rz},eab:function(){return tu},OI3:function(){return D.O},Fp7:function(){return eM},_sB:function(){return tl},YQQ:function(){return tp},Ip$:function(){return tc},gWQ:function(){return th.g},J69:function(){return td},ry_:function(){return ty},VV$:function(){return eD},LTh:function(){return tb},VdP:function(){return tk},wQq:function(){return tN},Gi7:function(){return tv},p_:function(){return nW},dC7:function(){return $.d},rq4:function(){return tx},SJ_:function(){return tw},W76:function(){return e8.W},KOy:function(){return eC},Quu:function(){return tT},lfX:function(){return tS.l},iUs:function(){return tm},JpU:function(){return tI},op:function(){return u.op},N2O:function(){return t_},vku:function(){return tE},pNR:function(){return tA},koy:function(){return tM},t1L:function(){return tD},lGY:function(){return t$},d_R:function(){return tB},sQ3:function(){return e$.s},AL3:function(){return tO.A},S0v:function(){return tR.S},WVs:function(){return tC},$gW:function(){return tV},VT$:function(){return tP},N89:function(){return tL},TN_:function(){return tz},wzB:function(){return tH},nGf:function(){return tj},ruB:function(){return tK},LGj:function(){return tX},w6H:function(){return tZ},kwC:function(){return tQ.k},M25:function(){return tY},UYe:function(){return tJ.U},btT:function(){return t0.b},XLQ:function(){return I.X},GYS:function(){return t1},SDf:function(){return t2},diP:function(){return t3},sx7:function(){return t6},mG2:function(){return t4},QEs:function(){return nf},NMM:function(){return t5},bp0:function(){return t8},iD$:function(){return eF.i},snQ:function(){return nG},zcT:function(){return to},U8D:function(){return t7},U_I:function(){return t9},ODp:function(){return nt},XD2:function(){return F.X},Xxe:function(){return nn},tdS:function(){return rV},O$l:function(){return nr},R_K:function(){return na},tPi:function(){return B},jZU:function(){return ns},SmN:function(){return ni},CnO:function(){return no},p0P:function(){return nu},XAC:function(){return nl},Wvh:function(){return e7},fBT:function(){return tF},rVs:function(){return rW},ers:function(){return nq},uN7:function(){return rC},Vl2:function(){return nd},_b3:function(){return eB._},h62:function(){return eO.h},$i:function(){return nm},L9e:function(){return ng},knu:function(){return ny},Nbs:function(){return nb.N},NXj:function(){return nk},Z_8:function(){return rU},luU:function(){return te.l},Smz:function(){return eR.S},ORZ:function(){return nN},AEp:function(){return O},XeE:function(){return nv.X},RRF:function(){return nw},odF:function(){return nT},wOQ:function(){return nS.w},yXz:function(){return nI},Bfx:function(){return n_},xZs:function(){return nE},Gg6:function(){return eW},hg7:function(){return nA},p4s:function(){return nz.p},Xu6:function(){return nM},Two:function(){return nD},pUJ:function(){return n$},HHK:function(){return nF},GaM:function(){return nB},VD$:function(){return nO},arb:function(){return ex},itS:function(){return nV},lls:function(){return tf},P84:function(){return ew.P}});var r={};n.r(r),n.d(r,{conv2d:function(){return n0},depthwiseConv2d:function(){return n3},matMul:function(){return n6}});var a=n(6235),s=n(196),i=n(9121),o=n(3740),u=n(2668);let l=(0,u.op)({acos_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","acos");return s.BV.runKernel(i.VGw,{x:t})}}),p=(0,u.op)({acosh_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","acosh");return s.BV.runKernel(i.SpW,{x:t})}});var c=n(6407),h=n(569);let d=(0,u.op)({addN_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){h.hu(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),h.hu(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);let t=e.map((e,t)=>(0,o._1)(e,`tensors${t}`,"addN")),n=t[0];return t.forEach(e=>{if(e.dtype!==n.dtype)throw Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(e=>{if(!h.cO(e.shape,n.shape))throw Error("All tensors passed to tf.addN() must have the same shape")}),s.BV.runKernel(i.Xze,t)}}),f=(0,u.op)({all_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let r=(0,o._1)(e,"x","all","bool");return s.BV.runKernel(i.oT6,{x:r},{axis:t,keepDims:n})}}),m=(0,u.op)({any_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let r=(0,o._1)(e,"x","any","bool");return s.BV.runKernel(i.IKK,{x:r},{axis:t,keepDims:n})}}),g=(0,u.op)({argMax_:/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){let n=(0,o._1)(e,"x","argMax");return s.BV.runKernel(i.sJF,{x:n},{axis:t})}}),y=(0,u.op)({argMin_:/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){let n=(0,o._1)(e,"x","argMin");return s.BV.runKernel(i.aJk,{x:n},{axis:t})}}),b=(0,u.op)({asin_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","asin");return s.BV.runKernel(i.M2y,{x:t})}}),k=(0,u.op)({asinh_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","asinh");return s.BV.runKernel(i.qw7,{x:t})}}),N=(0,u.op)({atan_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","atan");return s.BV.runKernel(i.jMg,{x:t})}});var v=n(747);let x=(0,u.op)({atan2_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","atan2"),r=(0,o._1)(t,"b","atan2");[n,r]=(0,v.makeTypesMatch)(n,r);let a={a:n,b:r};return s.BV.runKernel(i.QCc,a)}}),w=(0,u.op)({atanh_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","atanh");return s.BV.runKernel(i.Oyi,{x:t})}});var T=n(2271),S=n(2582),I=n(4968);let _=(0,u.op)({avgPool_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a){let u=(0,o._1)(e,"x","avgPool","float32");h.hu(S.jT(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let l=u,p=!1;3===u.rank&&(p=!0,l=(0,I.X)(u,[1,u.shape[0],u.shape[1],u.shape[2]])),h.hu(4===l.rank,()=>`Error in avgPool: x must be rank 4 but got rank ${l.rank}.`),S.m("avgPool",r,a);let c={x:l},d=s.BV.runKernel(i.JhU,c,{filterSize:t,strides:n,pad:r,dimRoundingMode:a});return(d=(0,T.p)(d,u.dtype),p)?(0,I.X)(d,[d.shape[1],d.shape[2],d.shape[3]]):d}}),E=(0,u.op)({avgPool3d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,u="NDHWC"){let l=(0,o._1)(e,"x","avgPool3d","float32"),p=l,c=!1;4===l.rank&&(c=!0,p=(0,I.X)(l,[1,l.shape[0],l.shape[1],l.shape[2],l.shape[3]])),h.hu(5===p.rank,()=>`Error in avgPool3d: x must be rank 5 but got rank ${p.rank}.`),h.hu("NDHWC"===u,()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${u}`),(0,S.m)("avgPool3d",r,a);let d={x:p},f=s.BV.runKernel(i._k9,d,{filterSize:t,strides:n,pad:r,dimRoundingMode:a,dataFormat:u});return(f=(0,T.p)(f,p.dtype),c)?(0,I.X)(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}});var A=n(8723);let M=(0,u.op)({concat_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){(0,h.hu)(e.length>=1,()=>"Pass at least one tensor to concat");let n=(0,o.sI)(e,"tensors","concat","string_or_numeric");return("complex64"===n[0].dtype&&n.forEach(e=>{if("complex64"!==e.dtype)throw Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${e.dtype}. `)}),1===n.length)?(0,A.d)(n[0]):s.BV.runKernel(i.Eh3,n,{axis:t})}});var D=n(8687),$=n(4841),F=n(625);let B=(0,u.op)({slice_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","slice","string_or_numeric");if(0===r.rank)throw Error("Slicing scalar is not possible");return s.BV.runKernel(i.p2w,{x:r},{begin:t,size:n})}}),O=(0,u.op)({tanh_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","tanh","float32");return s.BV.runKernel(i.MIZ,{x:t})}}),R=(0,u.op)({basicLSTMCell_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,s){let i=(0,o._1)(e,"forgetBias","basicLSTMCell"),u=(0,o._1)(t,"lstmKernel","basicLSTMCell"),l=(0,o._1)(n,"lstmBias","basicLSTMCell"),p=(0,o._1)(r,"data","basicLSTMCell"),h=(0,o._1)(a,"c","basicLSTMCell"),d=(0,o._1)(s,"h","basicLSTMCell"),f=M([p,d],1),m=(0,D.O)(f,u),g=(0,c.I)(m,l),y=g.shape[0],b=g.shape[1]/4,k=[y,b],N=B(g,[0,0],k),v=B(g,[0,b],k),x=B(g,[0,2*b],k),w=B(g,[0,3*b],k),T=(0,c.I)((0,$.d)((0,F.X)(N),O(v)),(0,$.d)(h,(0,F.X)((0,c.I)(i,x)))),S=(0,$.d)(O(T),(0,F.X)(w));return[T,S]}}),C=(0,u.op)({batchToSpaceND_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","batchToSpaceND"),a=t.reduce((e,t)=>e*t);return h.hu(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),h.hu(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),h.hu(r.shape[0]%a==0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${a}`),s.BV.runKernel(i.zws,{x:r},{blockShape:t,crops:n})}}),V=(0,u.op)({batchNorm_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,u){var l;null==u&&(u=.001);let p=(0,o._1)(e,"x","batchNorm"),c=(0,o._1)(t,"mean","batchNorm"),d=(0,o._1)(n,"variance","batchNorm"),f;null!=a&&(f=(0,o._1)(a,"scale","batchNorm"));let m;null!=r&&(m=(0,o._1)(r,"offset","batchNorm")),h.hu(c.rank===d.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),h.hu(null==m||c.rank===m.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),h.hu(null==f||c.rank===f.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");let g,y=0===p.rank||1===p.rank?(0,I.X)(p,[1,1,1,p.size]):2===p.rank?(0,I.X)(p,[1,1,p.shape[0],p.shape[1]]):3===p.rank?(0,I.X)(p,[1,p.shape[0],p.shape[1],p.shape[2]]):p,b={x:y,scale:f,offset:m,mean:c,variance:d},k={varianceEpsilon:u},N=s.BV.runKernel(i.sHE,b,k);return(0,I.X)(N,p.shape)}}),P=(0,u.op)({batchNorm2d_:function(e,t,n,r,a,s){let i=(0,o._1)(e,"x","batchNorm"),u=(0,o._1)(t,"mean","batchNorm"),l=(0,o._1)(n,"variance","batchNorm"),p;null!=a&&(p=(0,o._1)(a,"scale","batchNorm"));let c;return null!=r&&(c=(0,o._1)(r,"offset","batchNorm")),h.hu(2===i.rank,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${i.rank}.`),h.hu(2===u.rank||1===u.rank,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${u.rank}.`),h.hu(2===l.rank||1===l.rank,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${l.rank}.`),null!=p&&h.hu(2===p.rank||1===p.rank,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${p.rank}.`),null!=c&&h.hu(2===c.rank||1===c.rank,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${c.rank}.`),V(i,u,l,c,p,s)}}),L=(0,u.op)({batchNorm3d_:function(e,t,n,r,a,s){let i=(0,o._1)(e,"x","batchNorm"),u=(0,o._1)(t,"mean","batchNorm"),l=(0,o._1)(n,"variance","batchNorm"),p;null!=a&&(p=(0,o._1)(a,"scale","batchNorm"));let c;return null!=r&&(c=(0,o._1)(r,"offset","batchNorm")),h.hu(3===i.rank,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${i.rank}.`),h.hu(3===u.rank||1===u.rank,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${u.rank}.`),h.hu(3===l.rank||1===l.rank,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${l.rank}.`),null!=p&&h.hu(3===p.rank||1===p.rank,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${p.rank}.`),null!=c&&h.hu(3===c.rank||1===c.rank,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${c.rank}.`),V(i,u,l,c,p,s)}}),z=(0,u.op)({batchNorm4d_:function(e,t,n,r,a,s){let i=(0,o._1)(e,"x","batchNorm"),u=(0,o._1)(t,"mean","batchNorm"),l=(0,o._1)(n,"variance","batchNorm"),p;null!=a&&(p=(0,o._1)(a,"scale","batchNorm"));let c;return null!=r&&(c=(0,o._1)(r,"offset","batchNorm")),h.hu(4===i.rank,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${i.rank}.`),h.hu(4===u.rank||1===u.rank,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${u.rank}.`),h.hu(4===l.rank||1===l.rank,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${l.rank}.`),null!=p&&h.hu(4===p.rank||1===p.rank,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${p.rank}.`),null!=c&&h.hu(4===c.rank||1===c.rank,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${c.rank}.`),V(i,u,l,c,p,s)}}),W=(0,u.op)({bincount_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","bincount"),a=(0,o._1)(t,"weights","bincount");return h.hu("int32"===r.dtype,()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),h.hu(n>=0,()=>`size must be non-negative, but got ${n}.`),h.hu(a.size===r.size||0===a.size,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${a.shape}.`),s.BV.runKernel(i.zvY,{x:r,weights:a},{size:n})}}),U=(0,u.op)({broadcastArgs_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"s0","broadcastArgs","int32"),r=(0,o._1)(t,"s1","broadcastArgs","int32");if(1!==n.rank)throw Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(1!==r.rank)throw Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${r.rank}`);return s.BV.runKernel(i.eEB,{s0:n,s1:r})}}),G=(0,u.op)({broadcastTo_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"broadcastTo","x"),r=n.shape;if((0,h.Mu)(t),t.length<n.rank)throw Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){let a=n.shape.slice();for(;a.length<t.length;)a.unshift(1);n=(0,I.X)(n,a)}let u=n.shape,l=Array.from(t);for(let p=t.length-1;p>=0;p--)if(u[p]===t[p])l[p]=1;else if(1!==n.shape[p])throw Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);let c=l.map((e,t)=>e>1?t:-1).filter(e=>e>=0);if(0===c.length)return(0,A.d)(n);let d={x:n};return s.BV.runKernel(i.n9L,d,{reps:l})}});var q=n(2657);let H=(0,u.op)({ceil_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","ceil","float32");return s.BV.runKernel(i.gJX,{x:t})}});var j=n(4006);let K=(0,u.op)({clipByValue_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","clipByValue");return(h.hu(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)?(0,j.h)(r.shape,t,r.dtype):s.BV.runKernel(i.xnO,{x:r},{clipValueMin:t,clipValueMax:n})}});var X=n(1661);let Z=(0,u.op)({concat1d_:function(e){return M(e,0)}});function Q(e,t){return M(e,t)}let Y=(0,u.op)({concat2d_:Q});function J(e,t){return M(e,t)}let ee=(0,u.op)({concat3d_:J});function et(e,t){return M(e,t)}let en=(0,u.op)({concat4d_:et}),er=(0,u.op)({conv2d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a="NHWC",u=[1,1],l){let p=(0,o._1)(e,"x","conv2d","float32"),c=(0,o._1)(t,"filter","conv2d","float32"),d=p,f=!1;3===p.rank&&(f=!0,d=(0,I.X)(p,[1,p.shape[0],p.shape[1],p.shape[2]])),h.hu(4===d.rank,()=>`Error in conv2d: input must be rank 4, but got rank ${d.rank}.`),h.hu(4===c.rank,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),S.m("conv2d",r,l);let m="NHWC"===a?d.shape[3]:d.shape[1];h.hu(m===c.shape[2],()=>`Error in conv2d: depth of input (${m}) must match input depth for filter ${c.shape[2]}.`),h.hu(S.jT(n,u),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${u}'`);let g={x:d,filter:c},y=s.BV.runKernel(i.mhS,g,{strides:n,pad:r,dataFormat:a,dilations:u,dimRoundingMode:l});return f?(0,I.X)(y,[y.shape[1],y.shape[2],y.shape[3]]):y}}),ea=(0,u.op)({conv1d_:function(e,t,n,r,a="NWC",s=1,i){let u=(0,o._1)(e,"x","conv1d"),l=(0,o._1)(t,"filter","conv1d"),p=u,c=!1;2===u.rank&&(c=!0,p=(0,I.X)(u,[1,u.shape[0],u.shape[1]])),h.hu(3===p.rank,()=>`Error in conv1d: input must be rank 3, but got rank ${p.rank}.`),h.hu(3===l.rank,()=>`Error in conv1d: filter must be rank 3, but got rank ${l.rank}.`),S.m("conv1d",r,i),h.hu(p.shape[2]===l.shape[1],()=>`Error in conv1d: depth of input (${p.shape[2]}) must match input depth for filter ${l.shape[1]}.`),h.hu(S.jT(n,s),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${s}'`),h.hu("NWC"===a,()=>`Error in conv1d: got dataFormat of ${a} but only NWC is currently supported.`);let d=(0,I.X)(l,[1,l.shape[0],l.shape[1],l.shape[2]]),f=(0,I.X)(p,[p.shape[0],1,p.shape[1],p.shape[2]]),m=er(f,d,[1,n],r,"NHWC",[1,s],i);return c?(0,I.X)(m,[m.shape[2],m.shape[3]]):(0,I.X)(m,[m.shape[0],m.shape[2],m.shape[3]])}}),es=(0,u.op)({conv2DBackpropInput_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,o="NHWC",u){h.hu(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let l=e,p=t,c=!1;3===t.rank&&(c=!0,p=(0,I.X)(t,[1,t.shape[0],t.shape[1],t.shape[2]]),l=[1,e[0],e[1],e[2]]),h.hu(4===l.length,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${l.length}.`),h.hu(4===p.rank,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${p.rank}`),h.hu(4===n.rank,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);let d="NHWC"===o?l[3]:l[1],f="NHWC"===o?p.shape[3]:p.shape[1];h.hu(d===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${d}) must match input depth for filter ${n.shape[2]}.`),h.hu(f===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${f}) must match output depth for filter ${n.shape[3]}.`),S.m("conv2dDerInput",a,u);let m={dy:p,filter:n},g={strides:r,pad:a,dataFormat:o,dimRoundingMode:u,inputShape:l},y=s.BV.runKernel(i.wm,m,g);return c?(0,I.X)(y,[y.shape[1],y.shape[2],y.shape[3]]):y}}),ei=(0,u.op)({conv2dTranspose_:function(e,t,n,r,a,s){let i=(0,o._1)(e,"x","conv2dTranspose"),u=(0,o._1)(t,"filter","conv2dTranspose");return es(n,i,u,r,a,"NHWC",s)}}),eo=(0,u.op)({conv3d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a="NDHWC",u=[1,1,1]){let l=(0,o._1)(e,"x","conv3d"),p=(0,o._1)(t,"filter","conv3d"),c=l,d=!1;4===l.rank&&(d=!0,c=(0,I.X)(l,[1,l.shape[0],l.shape[1],l.shape[2],l.shape[3]])),h.hu(5===c.rank,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),h.hu(5===p.rank,()=>`Error in conv3d: filter must be rank 5, but got rank ${p.rank}.`),h.hu(c.shape[4]===p.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${p.shape[3]}.`),h.hu((0,S.jT)(n,u),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${u}'`),h.hu("NDHWC"===a,()=>`Error in conv3d: got dataFormat of ${a} but only NDHWC is currently supported.`);let f={x:c,filter:p},m=s.BV.runKernel(i.x12,f,{strides:n,pad:r,dataFormat:a,dilations:u});return d?(0,I.X)(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}}),eu=(0,u.op)({conv3DBackpropInput_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a){h.hu(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let o=e,u=t,l=!1;4===t.rank&&(l=!0,u=(0,I.X)(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),o=[1,e[0],e[1],e[2],e[3]]);let p=o[4],c=u.shape[4];h.hu(5===o.length,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${o.length}.`),h.hu(5===u.rank,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${u.rank}`),h.hu(5===n.rank,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),h.hu(p===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${p}) must match input depth for filter ${n.shape[3]}.`),h.hu(c===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${c}) must match output depth for filter ${n.shape[4]}.`);let d={dy:u,filter:n},f={pad:a,strides:r,inputShape:o},m=s.BV.runKernel(i.ik2,d,f);return l?(0,I.X)(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}}),el=(0,u.op)({conv3dTranspose_:function(e,t,n,r,a){let s=(0,o._1)(e,"x","conv3dTranspose"),i=(0,o._1)(t,"filter","conv3dTranspose");return eu(n,s,i,r,a)}}),ep=(0,u.op)({cos_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","cos","float32");return s.BV.runKernel(i.mc4,{x:t})}}),ec=(0,u.op)({cosh_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","cosh","float32");return s.BV.runKernel(i.TR1,{x:t})}}),eh=(0,u.op)({cumprod_:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0,n=!1,r=!1){let a=(0,o._1)(e,"x","cumprod");return s.BV.runKernel(i.Byc,{x:a},{axis:t,exclusive:n,reverse:r})}}),ed=(0,u.op)({cumsum_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0,n=!1,r=!1){let a=(0,o._1)(e,"x","cumsum");return s.BV.runKernel(i.iHb,{x:a},{axis:t,exclusive:n,reverse:r})}}),ef=(0,u.op)({denseBincount_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=!1){let a=(0,o._1)(e,"x","denseBincount"),u=(0,o._1)(t,"weights","denseBincount");return h.hu("int32"===a.dtype,()=>`Error in denseBincount: input dtype must be int32, but got ${a.dtype}`),h.hu(a.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${a.rank}.`),h.hu(n>=0,()=>`size must be non-negative, but got ${n}.`),h.hu(u.size===a.size||0===u.size,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${a.shape}, weights shape: ${u.shape}.`),s.BV.runKernel(i.QRR,{x:a,weights:u},{size:n,binaryOutput:r})}}),em=(0,u.op)({depthToSpace_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n="NHWC"){let r=(0,o._1)(e,"x","depthToSpace","float32"),a="NHWC"===n?r.shape[1]:r.shape[2],u="NHWC"===n?r.shape[2]:r.shape[3],l="NHWC"===n?r.shape[3]:r.shape[1];return h.hu(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),h.hu(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t}  for depthToSpace with input shape
    ${r.shape}`),h.hu(u*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${u} and ${t} for depthToSpace with input shape
        ${r.shape}`),h.hu(l%(t*t)==0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${l} for depthToSpace with input shape ${r.shape}`),s.BV.runKernel(i.T0n,{x:r},{blockSize:t,dataFormat:n})}}),eg=(0,u.op)({depthwiseConv2d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a="NHWC",u=[1,1],l){let p=(0,o._1)(e,"x","depthwiseConv2d","float32"),c=(0,o._1)(t,"filter","depthwiseConv2d","float32"),d=p,f=!1;3===p.rank&&(f=!0,d=(0,I.X)(p,[1,p.shape[0],p.shape[1],p.shape[2]])),h.hu(4===d.rank,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${d.rank}.`),h.hu(4===c.rank,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);let m="NHWC"===a?d.shape[3]:d.shape[1];h.hu(m===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${m}) must match the inChannels dimension in filter ${c.shape[2]}.`),S.m("depthwiseConv2d",r,l);let g={x:d,filter:c},y=s.BV.runKernel(i.cie,g,{strides:n,pad:r,dataFormat:a,dilations:u,dimRoundingMode:l});return f?(0,I.X)(y,[y.shape[1],y.shape[2],y.shape[3]]):y}}),ey=(0,u.op)({diag_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","diag");return s.BV.runKernel(i.$w,{x:t})}}),eb=(0,u.op)({dilation2d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a=[1,1],u="NHWC"){let l=(0,o._1)(e,"x","dilation2d"),p=(0,o._1)(t,"filter","dilation2d");h.hu(3===l.rank||4===l.rank,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${l.rank}.`),h.hu(3===p.rank,()=>`Error in dilation2d: filter must be rank 3, but got rank ${p.rank}.`),h.hu("NHWC"===u,()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${u}`);let c=l,d=!1;3===l.rank&&(c=(0,I.X)(l,[1,l.shape[0],l.shape[1],l.shape[2]]),d=!0);let f={x:c,filter:p},m=s.BV.runKernel(i.p4S,f,{strides:n,pad:r,dilations:a});return d?(0,I.X)(m,[m.shape[1],m.shape[2],m.shape[3]]):m}});var ek=n(1274),eN=n(2200);let ev=(0,u.op)({equal_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","equal","string_or_numeric"),r=(0,o._1)(t,"b","equal","string_or_numeric");[n,r]=(0,v.makeTypesMatch)(n,r),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i.hdR,a)}}),ex=(0,u.op)({where_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(t,"a","where"),a=(0,o._1)(n,"b","where"),u=(0,o._1)(e,"condition","where","bool"),l=(0,eN.assertAndGetBroadcastShape)((0,eN.assertAndGetBroadcastShape)(u.shape,r.shape),a.shape),p=G(u,l),c=G(r,l),h=G(a,l);return s.BV.runKernel(i.PhF,{condition:p,t:c,e:h})}});var ew=n(6577);let eT=(0,u.op)({divNoNan_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","div"),r=(0,o._1)(t,"b","div");[n,r]=(0,v.makeTypesMatch)(n,r);let a=(0,ek.h)(n,r),s=(0,ew.P)(a),i=ev(r,s);return ex(i,s,a)}}),eS=(0,u.op)({dot_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"t1","dot"),r=(0,o._1)(t,"t2","dot");h.hu((1===n.rank||2===n.rank)&&(1===r.rank||2===r.rank),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);let a=1===n.rank?n.size:n.shape[1],s=1===r.rank?r.size:r.shape[0];if(h.hu(a===s,()=>`Error in dot: inner dimensions of inputs must match, but got ${a} and ${s}.`),1===n.rank&&1===r.rank){let i=(0,I.X)(n,[1,-1]),u=(0,I.X)(r,[-1,1]),l=(0,D.O)(i,u);return(0,I.X)(l,[])}if(1===n.rank&&2===r.rank){let p=(0,I.X)(n,[1,-1]),c=(0,I.X)(r,[r.shape[0],r.shape[1]]),d=(0,D.O)(p,c);return(0,I.X)(d,[d.size])}if(2===n.rank&&1===r.rank){let f=(0,I.X)(r,[-1,1]),m=(0,D.O)(n,f);return(0,I.X)(m,[m.size])}{let g=(0,I.X)(r,[r.shape[0],r.shape[1]]),y=(0,D.O)(n,g);return y}}}),eI=(0,u.op)({einsum_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,...t){let n=t.map((e,t)=>(0,o._1)(e,`tensors${t}`,"einsum"));return s.BV.runKernel(i.$g6,n,{equation:e})}});var e_=n(3233);let eE=(0,u.op)({erf_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","erf");h.hu("int32"===t.dtype||"float32"===t.dtype,()=>"Input dtype must be `int32` or `float32`."),"int32"===t.dtype&&(t=(0,T.p)(t,"float32"));let n={x:t};return s.BV.runKernel(i.Omj,n)}});var eA=n(3591);let eM=(0,u.op)({max_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let r=(0,o._1)(e,"x","max");return s.BV.runKernel(i.YoZ,{x:r},{reductionIndices:t,keepDims:n})}}),eD=(0,u.op)({min_:/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let r=(0,o._1)(e,"x","min");return s.BV.runKernel(i.c17,{x:r},{axis:t,keepDims:n})}});var e$=n(3453),eF=n(9494),eB=n(3261),eO=n(248),eR=n(5475);let eC=(0,u.op)({norm_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t="euclidean",n=null,r=!1){e=(0,o._1)(e,"x","norm");let s=function e(t,n,r=null){if(0===t.rank)return(0,a.W)(t);if(1!==t.rank&&null===r)return e((0,I.X)(t,[-1]),n,r);if(1===t.rank||"number"==typeof r||Array.isArray(r)&&1===r.length){if(1===n)return(0,eR.S)((0,a.W)(t),r);if(n===1/0)return eM((0,a.W)(t),r);if(n===-1/0)return eD((0,a.W)(t),r);if("euclidean"===n||2===n)return(0,eB._)((0,eR.S)((0,e$.s)((0,a.W)(t),(0,eF.i)(2,"int32")),r));throw Error(`Error in norm: invalid ord value: ${n}`)}if(Array.isArray(r)&&2===r.length){if(1===n)return eM((0,eR.S)((0,a.W)(t),r[0]),r[1]-1);if(n===1/0)return eM((0,eR.S)((0,a.W)(t),r[1]),r[0]);if(n===-1/0)return eD((0,eR.S)((0,a.W)(t),r[1]),r[0]);if("fro"===n||"euclidean"===n)return(0,eB._)((0,eR.S)((0,eO.h)(t),r));throw Error(`Error in norm: invalid ord value: ${n}`)}throw Error(`Error in norm: invalid axis: ${r}`)}(e,t,n),i=s.shape;if(r){let u=(0,h.EC)(n,e.shape);i=eA.rv(s.shape,u)}return(0,I.X)(s,i)}}),eV=(0,u.op)({euclideanNorm_:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){return eC(e,"euclidean",t,n)}}),eP=(0,u.op)({exp_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","exp");return s.BV.runKernel(i.NEP,{x:t})}}),eL=(0,u.op)({expandDims_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){let n=(0,o._1)(e,"x","expandDims","string_or_numeric");return h.hu(t<=n.rank,()=>"Axis must be <= rank of the tensor"),s.BV.runKernel(i.YFo,{input:n},{dim:t})}}),ez=(0,u.op)({expm1_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","expm1");return s.BV.runKernel(i.Y0y,{x:t})}}),eW=(0,u.op)({tile_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"x","tile","string_or_numeric");return h.hu(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`),s.BV.runKernel(i.n9L,{x:n},{reps:t})}}),eU=(0,u.op)({eye_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r="float32"){null==t&&(t=e);let a=(0,q.f)([e,t],r),s=e<=t?e:t;for(let i=0;i<s;++i)a.set(1,i,i);let o=(0,I.X)(a.toTensor(),[e,t]);if(null==n)return o;if(1===n.length)return eW(eL(o,0),[n[0],1,1]);if(2===n.length)return eW(eL(eL(o,0),0),[n[0],n[1],1,1]);if(3===n.length)return eW(eL(eL(eL(o,0),0),0),[n[0],n[1],n[2],1,1]);throw Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}}),eG=(0,u.op)({floor_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","floor","float32");return s.BV.runKernel(i.OR,{x:t})}});var eq=n(9165);let eH=(0,u.op)({gather_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=0,r=0){let a=(0,o._1)(e,"x","gather"),u=(0,o._1)(t,"indices","gather","int32");return s.BV.runKernel(i.qi_,{x:a,indices:u},{axis:n,batchDims:r})}}),ej=(0,u.op)({greater_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","greater","string_or_numeric"),r=(0,o._1)(t,"b","greater","string_or_numeric");[n,r]=(0,v.makeTypesMatch)(n,r),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i.iZT,a)}}),eK=(0,u.op)({greaterEqual_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","greaterEqual","string_or_numeric"),r=(0,o._1)(t,"b","greaterEqual","string_or_numeric");[n,r]=(0,v.makeTypesMatch)(n,r),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i.Acj,a)}});var eX=n(4386);let eZ=(0,u.op)({isFinite_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","isFinite");return s.BV.runKernel(i.avt,{x:t})}}),eQ=(0,u.op)({isInf_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","isInf");return s.BV.runKernel(i.iWB,{x:t})}}),eY=(0,u.op)({isNaN_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","isNaN");return s.BV.runKernel(i.r7n,{x:t})}});var eJ=n(9133);let e0=(0,u.op)({less_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","less","string_or_numeric"),r=(0,o._1)(t,"b","less","string_or_numeric");[n,r]=(0,v.makeTypesMatch)(n,r),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i.vtC,a)}}),e1=(0,u.op)({lessEqual_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","lessEqual","string_or_numeric"),r=(0,o._1)(t,"b","lessEqual","string_or_numeric");[n,r]=(0,v.makeTypesMatch)(n,r),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i.CAk,a)}});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function e2(e,t,n){if(n<=0)throw Error("The number of values should be positive.");return s.BV.runKernel(i.e7N,{},{start:e,stop:t,num:n})}let e3=(0,u.op)({localResponseNormalization_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=5,n=1,r=1,a=.5){let u=(0,o._1)(e,"x","localResponseNormalization");h.hu(4===u.rank||3===u.rank,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${u.rank}.`),h.hu(h.GN(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let l=u,p=!1;3===u.rank&&(p=!0,l=(0,I.X)(u,[1,u.shape[0],u.shape[1],u.shape[2]]));let c={x:l},d=s.BV.runKernel(i.eZ0,c,{depthRadius:t,bias:n,alpha:r,beta:a});return p?(0,I.X)(d,[d.shape[1],d.shape[2],d.shape[3]]):d}}),e6=(0,u.op)({log_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","log","float32");return s.BV.runKernel(i.ZbH,{x:t})}}),e4=(0,u.op)({log1p_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","log1p");return s.BV.runKernel(i.kU,{x:t})}});var e5=n(633),e8=n(7370);let e7=(0,u.op)({softplus_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","softplus");return s.BV.runKernel(i.MRv,{x:t})}}),e9=(0,u.op)({logSigmoid_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","logSigmoid"),n=(0,e5.cb)(e=>{let t=(0,e8.W)(e7((0,e8.W)(e))),n=t=>{let n=(0,$.d)(t,(0,F.X)((0,e8.W)(e)));return n};return{value:t,gradFunc:n}});return n(t)}});var te=n(827);let tt=(0,u.op)({logSoftmax_:/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=-1){let n=(0,o._1)(e,"logits","logSoftmax");if(-1===t&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);let r=(0,e5.cb)((e,n)=>{let r=eM(e,t,!0),a=(0,te.l)(e,r),s=(0,te.l)((0,T.p)(a,"float32"),e6((0,eR.S)(eP(a),t,!0)));n([s]);let i=(e,n)=>{let[r]=n,a=eP(r);return(0,te.l)(e,(0,$.d)((0,eR.S)(e,t,!0),a))};return{value:s,gradFunc:i}});return r(n)}}),tn=(0,u.op)({logSumExp_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let r=(0,o._1)(e,"x","logSumExp"),a=(0,h.EC)(t,r.shape),s=eM(r,a,!0),i=(0,te.l)(r,s),u=eP(i),l=(0,eR.S)(u,a),p=e6(l),d=(0,c.I)((0,I.X)(s,p.shape),p);if(n){let f=(0,eA.rv)(d.shape,a);return(0,I.X)(d,f)}return d}}),tr=(0,u.op)({logicalAnd_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","logicalAnd","bool"),r=(0,o._1)(t,"b","logicalAnd","bool");return(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape),s.BV.runKernel(i.PYm,{a:n,b:r})}}),ta=(0,u.op)({logicalNot_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","logicalNot","bool");return s.BV.runKernel(i.VfG,{x:t})}}),ts=(0,u.op)({logicalOr_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","logicalOr","bool"),r=(0,o._1)(t,"b","logicalOr","bool");return(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape),s.BV.runKernel(i.MZg,{a:n,b:r})}}),ti=(0,u.op)({logicalXor_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","logicalXor","bool"),r=(0,o._1)(t,"b","logicalXor","bool");return(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape),tr(ts(e,t),ta(tr(e,t)))}}),to=(0,u.op)({searchSorted_:function(e,t,n="left"){let r=(0,o._1)(e,"sortedSequence","searchSorted"),a=(0,o._1)(t,"values","searchSorted"),u=r.shape[r.shape.length-1],l=a.shape[a.shape.length-1],p=(0,I.X)(r,[-1,u]),c=(0,I.X)(a,[-1,l]);if(p.rank<2)throw Error("Sorted input argument must be at least 2-dimensional");if(p.shape[0]!==c.shape[0])throw Error("Leading dimension of 'sortedSequence' and 'values' must match.");if((0,h.NA)(c.shape)>=2147483648)throw Error("values tensor size must less than 2147483648");if(p.shape[1]>=2147483648)throw Error(`trailing dim_size must less than 2147483648 for int32 output type, was ${p.shape[1]}`);return s.BV.runKernel(i.nr8,{sortedSequence:p,values:c},{side:n})}});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function tu(e,t){return to(e,t,"left")}let tl=(0,u.op)({maxPool_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a){let u=(0,o._1)(e,"x","maxPool"),l=u,p=!1;3===u.rank&&(p=!0,l=(0,I.X)(u,[1,u.shape[0],u.shape[1],u.shape[2]])),h.hu(4===l.rank,()=>`Error in maxPool: input must be rank 4 but got rank ${l.rank}.`),h.hu(S.jT(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),S.m("maxPool",r,a);let c={x:l},d=s.BV.runKernel(i.mTV,c,{filterSize:t,strides:n,pad:r,dimRoundingMode:a});return p?(0,I.X)(d,[d.shape[1],d.shape[2],d.shape[3]]):d}}),tp=(0,u.op)({maxPool3d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=[1,1,1],n,r,a,u="NDHWC"){let l=(0,o._1)(e,"x","maxPool3d"),p=l,c=!1;4===l.rank&&(c=!0,p=(0,I.X)(l,[1,l.shape[0],l.shape[1],l.shape[2],l.shape[3]])),h.hu(5===p.rank,()=>`Error in maxPool3d: x must be rank 5 but got rank ${p.rank}.`),h.hu("NDHWC"===u,()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${u}`),(0,S.m)("maxPool3d",r,a);let d={x:p},f=s.BV.runKernel(i.OAf,d,{filterSize:t,strides:n,pad:r,dimRoundingMode:a,dataFormat:u});return c?(0,I.X)(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}}),tc=(0,u.op)({maxPoolWithArgmax_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a=!1){let u=(0,o._1)(e,"x","maxPoolWithArgmax"),l=s.BV.runKernel(i.vFR,{x:u},{filterSize:t,strides:n,pad:r,includeBatchInIndex:a});return{result:l[0],indexes:l[1]}}});var th=n(632);let td=(0,u.op)({mean_:/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let r=(0,o._1)(e,"x","mean");return s.BV.runKernel(i.q2K,{x:r},{axis:t,keepDims:n})}});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function tf(e,t="float32"){if((0,h.Mu)(e),"complex64"===t){let n=tf(e,"float32"),r=tf(e,"float32");return(0,X.P)(n,r)}let a=(0,h.wT)((0,h.NA)(e),t);return s.BV.makeTensor(a,e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function tm(e,t="float32"){if((0,h.Mu)(e),"complex64"===t){let n=tm(e,"float32"),r=tf(e,"float32");return(0,X.P)(n,r)}let a=(0,h.p8)((0,h.NA)(e),t);return s.BV.makeTensor(a,e,t)}var tg=n(974);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function ty(e,t,{indexing:n="xy"}={}){if("xy"!==n&&"ij"!==n)throw TypeError(`${n} is not a valid third argument to meshgrid`);if(void 0===e)return[];let r=(0,o._1)(e,"x","meshgrid",e instanceof tg.es?e.dtype:"float32");if(void 0===t)return[r];let a=(0,o._1)(t,"y","meshgrid",t instanceof tg.es?t.dtype:"float32"),s=(0,h.NA)(r.shape),i=(0,h.NA)(a.shape);return"xy"===n?(r=(0,I.X)(r,[1,-1]),a=(0,I.X)(a,[-1,1]),[(0,D.O)(tm([i,1],r.dtype),r),(0,D.O)(a,tm([1,s],a.dtype)),]):(r=(0,I.X)(r,[-1,1]),a=(0,I.X)(a,[1,-1]),[(0,D.O)(r,tm([1,i],r.dtype)),(0,D.O)(tm([s,1],a.dtype),a),])}let tb=(0,u.op)({minimum_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","minimum"),r=(0,o._1)(t,"b","minimum");[n,r]=(0,v.makeTypesMatch)(n,r),"bool"===n.dtype&&(n=(0,T.p)(n,"int32"),r=(0,T.p)(r,"int32")),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i.q8u,a)}}),tk=(0,u.op)({mirrorPad_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){h.hu("reflect"===n||"symmetric"===n,()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);let r=(0,o._1)(e,"x","mirrorPad");if(0===r.rank)throw Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");h.hu(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);let a="reflect"===n?1:0;for(let u=0;u<r.rank;u++)h.hu(2===t[u].length,()=>"Invalid number of paddings. Must be length of 2 each."),h.hu(t[u][0]>=0&&t[u][0]<=r.shape[u]-a&&t[u][1]>=0&&t[u][1]<=r.shape[u]-a,()=>`Padding in dimension ${u} cannot be greater than or equal to ${r.shape[u]-a} or less than 0 for input of shape ${r.shape}`);return s.BV.runKernel(i.jQs,{x:r},{paddings:t,mode:n})}}),tN=(0,u.op)({mod_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","mod"),r=(0,o._1)(t,"b","mod");[n,r]=(0,v.makeTypesMatch)(n,r);let a={a:n,b:r};return s.BV.runKernel(i.Vbg,a)}}),tv=(0,u.op)({moments_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){e=(0,o._1)(e,"x","moments");let r=(0,h.EC)(t,e.shape),a=td(e,r,n),s=a.shape;n||(s=(0,eA.rv)(a.shape,r));let i=(0,eO.h)((0,te.l)((0,T.p)(e,"float32"),(0,I.X)(a,s))),u=td(i,r,n);return{mean:a,variance:u}}}),tx=(0,u.op)({multiRNNCell_:function(e,t,n,r){let a=(0,o._1)(t,"data","multiRNNCell"),s=(0,o.sI)(n,"c","multiRNNCell"),i=(0,o.sI)(r,"h","multiRNNCell"),u=a,l=[];for(let p=0;p<e.length;p++){let c=e[p](u,s[p],i[p]);l.push(c[0]),l.push(c[1]),u=c[1]}let h=[],d=[];for(let f=0;f<l.length;f+=2)h.push(l[f]),d.push(l[f+1]);return[h,d]}}),tw=(0,u.op)({multinomial_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=!1){let a=(0,o._1)(e,"logits","multinomial"),u=a.size,l=a.rank;if(u<2)throw Error(`Error in multinomial: you need at least 2 outcomes, but got ${u}.`);if(l>2)throw Error(`Rank of probabilities must be 1 or 2, but is ${l}`);n=n||Math.random();let p=1===l?(0,I.X)(a,[1,-1]):a,c={numSamples:t,seed:n,normalized:r},h=s.BV.runKernel(i.NZg,{logits:p},c);return 1===l?(0,I.X)(h,[h.size]):h}}),tT=(0,u.op)({notEqual_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","notEqual","string_or_numeric"),r=(0,o._1)(t,"b","notEqual","string_or_numeric");[n,r]=(0,v.makeTypesMatch)(n,r),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i.yQU,a)}});var tS=n(6708);let tI=(0,u.op)({onesLike_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","onesLike");return s.BV.runKernel(i.qWM,{x:t})}}),t_=(0,u.op)({outerProduct_:function(e,t){let n=(0,o._1)(e,"v1","outerProduct"),r=(0,o._1)(t,"v2","outerProduct");h.hu(1===n.rank&&1===r.rank,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${r.rank}.`);let a=(0,I.X)(n,[-1,1]),s=(0,I.X)(r,[1,-1]);return(0,D.O)(a,s)}}),tE=(0,u.op)({pad_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=0){let r=(0,o._1)(e,"x","pad");if(0===r.rank)throw Error("pad(scalar) is not defined. Pass non-scalar to pad");return s.BV.runKernel(i.lyA,{x:r},{paddings:t,constantValue:n})}}),tA=(0,u.op)({pad1d_:function(e,t,n=0){return(0,h.hu)(2===t.length,()=>"Invalid number of paddings. Must be length of 2."),tE(e,[t],n)}}),tM=(0,u.op)({pad2d_:function(e,t,n=0){return(0,h.hu)(2===t.length&&2===t[0].length&&2===t[1].length,()=>"Invalid number of paddings. Must be length of 2 each."),tE(e,t,n)}}),tD=(0,u.op)({pad3d_:function(e,t,n=0){return(0,h.hu)(3===t.length&&2===t[0].length&&2===t[1].length&&2===t[2].length,()=>"Invalid number of paddings. Must be length of 2 each."),tE(e,t,n)}}),t$=(0,u.op)({pad4d_:function(e,t,n=0){return(0,h.hu)(4===t.length&&2===t[0].length&&2===t[1].length&&2===t[2].length&&2===t[3].length,()=>"Invalid number of paddings. Must be length of 2 each."),tE(e,t,n)}}),tF=(0,u.op)({spaceToBatchND_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","spaceToBatchND");return h.hu(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),h.hu(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),h.hu(r.shape.reduce((e,r,a)=>a>0&&a<=t.length?e&&(r+n[a-1][0]+n[a-1][1])%t[a-1]==0:e,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`),s.BV.runKernel(i.TQc,{x:r},{blockShape:t,paddings:n})}}),tB=(0,u.op)({pool_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,s,i){null==a&&(a=[1,1]),null==s&&(s=1),0===r&&(r="valid");let u=(0,o._1)(e,"x","maxPool"),l=u,p=!1;3===u.rank&&(p=!0,l=(0,I.X)(u,[1,u.shape[0],u.shape[1],u.shape[2]])),h.hu(S.jT(s,a),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${s} and dilations '${a}'`);let c=S.Xw(l.shape,t,s,a,r),d=[c.dilationHeight,c.dilationWidth],f;f="same"===r?function(e,t){let n=e.map((e,n)=>e+(e-1)*(t[n]-1)),r=n.map(e=>e-1),a=r.map(e=>Math.floor(e/2)),s=r.map((e,t)=>e-a[t]);return r.map((e,t)=>[a[t],s[t]])}([c.filterHeight,c.filterWidth],d):[[0,0],[0,0]];let m=1===d[0]&&1===d[1],[g,y]=function(e,t,n){let r=n.map(e=>e[0]),a=n.map(e=>e[1]),s=e.concat(r,a),i=t.map((e,t)=>(e-s[t]%e)%e),o=a.map((e,t)=>e+i[t]),u=t.map((e,t)=>[r[t],o[t]]),l=t.map((e,t)=>[0,i[t]]);return[u,l]}([c.inHeight,c.inWidth],d,f),b=m?r:"valid",k=m?l:tF(l,d,g),N="avg"===n?()=>_(k,t,s,b,i):()=>tl(k,t,s,b,i),v=N(),x=m?v:C(v,d,y);return p?(0,I.X)(x,[x.shape[1],x.shape[2],x.shape[3]]):x}});var tO=n(8151),tR=n(9798);let tC=(0,u.op)({prod_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let r=(0,o._1)(e,"x","prod");"bool"===r.dtype&&(r=(0,T.p)(r,"int32"));let a={x:r};return s.BV.runKernel(i.DlI,a,{axis:t,keepDims:n})}}),tV=(0,u.op)({raggedGather_:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r){let a=e.map((e,t)=>(0,o._1)(e,`tensors${t}`,"raggedGather","int32")),u=(0,o._1)(t,"paramsDenseValues","raggedGather"),l=(0,o._1)(n,"indices","raggedGather","int32"),p=s.BV.runKernel(i.dDz,{paramsNestedSplits:a,paramsDenseValues:u,indices:l},{outputRaggedRank:r});return{outputNestedSplits:p.slice(0,p.length-1),outputDenseValues:p[p.length-1]}}}),tP=(0,u.op)({raggedRange_:/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"starts","raggedRange"),a=(0,o._1)(t,"limits","raggedRange",r.dtype),u=(0,o._1)(n,"deltas","raggedRange",r.dtype),l=s.BV.runKernel(i.CQl,{starts:r,limits:a,deltas:u});return{rtNestedSplits:l[0],rtDenseValues:l[1]}}}),tL=(0,u.op)({raggedTensorToTensor_:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a){let u=(0,o._1)(e,"shape","raggedTensorToTensor","int32"),l=(0,o._1)(t,"values","raggedTensorToTensor"),p=(0,o._1)(n,"defaultValue","raggedTensorToTensor",l.dtype),c=r.map((e,t)=>(0,o._1)(e,`tensors${t}`,"raggedTensorToTensor","int32"));return s.BV.runKernel(i.BiW,{shape:u,values:l,defaultValue:p,rowPartitionTensors:c},{rowPartitionTypes:a})}}),tz=(0,u.op)({rand_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){(0,h.Mu)(e);let r=(0,h.NA)(e),a=null;if(null==n||"float32"===n)a=new Float32Array(r);else if("int32"===n)a=new Int32Array(r);else if("bool"===n)a=new Uint8Array(r);else throw Error(`Unknown data type ${n}`);for(let i=0;i<r;i++)a[i]=t();return s.BV.makeTensor(a,e,n)}});var tW=n(6377);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class tU{constructor(e,t,n,r,a){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+2*this.stdDev,this.lower=this.mean-2*this.stdDev);let s=a||Math.random();this.random=tW.alea(s.toString())}nextValue(){if(!isNaN(this.nextVal)){let e=this.nextVal;return this.nextVal=NaN,e}let t,n,r=!1;for(;!r;){let a,s,i;do i=(a=2*this.random()-1)*a+(s=2*this.random()-1)*s;while(i>=1||0===i);let o=Math.sqrt(-2*Math.log(i)/i);t=this.mean+this.stdDev*a*o,n=this.mean+this.stdDev*s*o,(!this.truncated||this.isValidTruncated(t))&&(r=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(t)}convertValue(e){return null==this.dtype||"float32"===this.dtype?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}}class tG{constructor(e,t,n,r){this.alpha=e,this.beta=1/t,this.dtype=n;let a=r||Math.random();this.randu=tW.alea(a.toString()),this.randn=new tU(0,1,n,!1,this.randu()),e<1?this.d=e+2/3:this.d=e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,t,n,r,a,s;for(;;){do r=this.randn.nextValue(),s=1+this.c*r;while(s<=0);if(s*=s*s,t=1-.331*(e=r*r)*e,n=.5*e+this.d*(1-s+Math.log(s)),(a=this.randu())<t||Math.log(a)<n)break}return s=1/this.beta*this.d*s,this.alpha<1&&(s*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(s)}convertValue(e){return"float32"===this.dtype?e:Math.round(e)}}class tq{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>null==this.dtype||"float32"===this.dtype,this.min=e,this.range=t-e,this.dtype=n,null==r&&(r=Math.random()),"number"==typeof r&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=tW.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}let tH=(0,u.op)({randomGamma_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=1,r="float32",a){if((0,h.Mu)(e),null==n&&(n=1),null==r&&(r="float32"),"float32"!==r&&"int32"!==r)throw Error(`Unsupported data type ${r}`);let s=new tG(t,n,r,a),i=(0,q.f)(e,r);for(let o=0;o<i.values.length;o++)i.values[o]=s.nextValue();return i.toTensor()}}),tj=(0,u.op)({randomNormal_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0,n=1,r,a){if((0,h.Mu)(e),null!=r&&"bool"===r)throw Error(`Unsupported data type ${r}`);let s=new tU(t,n,r,!1,a),i=(0,q.f)(e,r);for(let o=0;o<i.values.length;o++)i.values[o]=s.nextValue();return i.toTensor()}}),tK=(0,u.op)({randomStandardNormal_:/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){if(null!=t&&"bool"===t)throw Error(`Unsupported data type ${t}`);return tj(e,0,1,t,n)}}),tX=(0,u.op)({randomUniform_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0,n=1,r="float32",a){(0,h.Mu)(e);let s=(0,q.f)(e,r),i=new tq(t,n,null,a);for(let o=0;o<s.values.length;o++)s.values[o]=i.nextValue();return s.toTensor()}});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function tZ(e,t,n=1,r="float32"){if(0===n)throw Error("Cannot have a step of zero");return s.BV.runKernel(i.e6w,{},{start:e,stop:t,step:n,dtype:r})}var tQ=n(766);let tY=(0,u.op)({reciprocal_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","reciprocal");return s.BV.runKernel(i.$HU,{x:t})}});var tJ=n(7409),t0=n(3582);let t1=(0,u.op)({reverse_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"x","reverse");return s.BV.runKernel(i.mKl,{x:n},{dims:t})}}),t2=(0,u.op)({reverse1d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","reverse");return h.hu(1===t.rank,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),t1(t,0)}}),t3=(0,u.op)({reverse2d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"x","reverse");return h.hu(2===n.rank,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),t1(n,t)}}),t6=(0,u.op)({reverse3d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"x","reverse");return h.hu(3===n.rank,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),t1(n,t)}}),t4=(0,u.op)({reverse4d_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"x","reverse");return h.hu(4===n.rank,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),t1(n,t)}}),t5=(0,u.op)({round_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","round");return s.BV.runKernel(i.e07,{x:t})}}),t8=(0,u.op)({rsqrt_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","rsqrt","float32");return s.BV.runKernel(i.bV0,{x:t})}}),t7=(0,u.op)({selu_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","selu");return s.BV.runKernel(i.oFR,{x:t})}}),t9=(0,u.op)({separableConv2d_:function(e,t,n,r,a,s=[1,1],i="NHWC"){let u=(0,o._1)(e,"x","separableConv2d"),l=(0,o._1)(t,"depthwiseFilter","separableConv2d"),p=(0,o._1)(n,"pointwiseFilter","separableConv2d"),c=u,d=!1;if(3===u.rank&&(d=!0,c=(0,I.X)(u,[1,u.shape[0],u.shape[1],u.shape[2]])),"NCHW"===i)throw Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");h.hu(4===c.rank,()=>`Error in separableConv2d: input must be rank 4, but got rank ${c.rank}.`),h.hu(4===l.rank,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${l.rank}.`),h.hu(4===p.rank,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${l.rank}.`),h.hu(1===p.shape[0],()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${p.shape[0]}.`),h.hu(1===p.shape[1],()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${p.shape[1]}.`);let f=l.shape[2],m=l.shape[3];h.hu(p.shape[2]===f*m,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*m}, but got ${p.shape[2]}.`);let g=eg(c,l,r,a,i,s),y=er(g,p,1,"valid",i);return d?(0,I.X)(y,[y.shape[1],y.shape[2],y.shape[3]]):y}});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function ne(e,t){let n=(0,o._1)(e,"x","setdiff1d"),r=(0,o._1)(t,"y","setdiff1d");h.hu(n.dtype===r.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${r.dtype}).`),h.hu(1===n.rank,()=>`x should be 1D tensor, but got x (${n.shape}).`),h.hu(1===r.rank,()=>`y should be 1D tensor, but got y (${r.shape}).`);let a=await n.data(),s=await r.data(),i=new Set(s),u=0;for(let l=0;l<a.length;l++)!i.has(a[l])&&u++;let p=new tg.YD([u],n.dtype),c=new tg.YD([u],"int32");for(let d=0,f=0;d<a.length;d++)!i.has(a[d])&&(p.values[f]=a[d],c.values[f]=d,f++);return[p.toTensor(),c.toTensor()]}let nt=ne,nn=(0,u.op)({sign_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","sign");return s.BV.runKernel(i.i5y,{x:t})}}),nr=(0,u.op)({sin_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","sin","float32");return s.BV.runKernel(i.RQH,{x:t})}}),na=(0,u.op)({sinh_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","sinh");return s.BV.runKernel(i.wYB,{x:t})}}),ns=(0,u.op)({slice1d_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","slice1d");return h.hu(1===r.rank,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),B(r,[t],[n])}}),ni=(0,u.op)({slice2d_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","slice2d");return h.hu(2===r.rank,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),B(r,t,n)}}),no=(0,u.op)({slice3d_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","slice3d");return h.hu(3===r.rank,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),B(r,t,n)}}),nu=(0,u.op)({slice4d_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","slice4d");return h.hu(4===r.rank,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),B(r,t,n)}}),nl=(0,u.op)({softmax_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=-1){let n=(0,o._1)(e,"logits","softmax","float32");if(-1===t&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);let r={dim:t};return s.BV.runKernel(i.Gcp,{logits:n},r)}}),np=(0,u.op)({fft_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){return(0,h.hu)("complex64"===e.dtype,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`),s.BV.runKernel(i.vwp,{input:e})}}),nc=(0,u.op)({ifft_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){return(0,h.hu)("complex64"===e.dtype,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`),s.BV.runKernel(i.Qg5,{input:e})}}),nh=(0,u.op)({irfft_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=e.shape[e.shape.length-1],n=e.size/t,r;if(t<=2){let a=(0,I.X)(e,[n,t]);r=nc(a)}else{let s=[n,2*(t-1)],i=(0,I.X)((0,tQ.k)(e),[n,t]),o=(0,I.X)((0,eX.a)(e),[n,t]),u=t1(B(i,[0,1],[n,t-2]),1),l=(0,$.d)(t1(B(o,[0,1],[n,t-2]),1),(0,eF.i)(-1)),p=M([i,u],1),c=M([o,l],1),h=(0,I.X)((0,X.P)(p,c),[s[0],s[1]]);r=nc(h)}if(r=(0,tQ.k)(r),3===e.rank&&0!==e.shape[0]){let d=r,f=e.shape[0];r=(0,I.X)(r,[f,r.shape[0]/f,r.shape[1]]),d.dispose()}return r}}),nd=(0,u.op)({split_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=0){let r=(0,o._1)(e,"x","split");return s.BV.runKernel(i.L8s,{x:r},{numOrSizeSplits:t,axis:n})}}),nf=(0,u.op)({rfft_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){(0,h.hu)("float32"===e.dtype,()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1],r=e.size/n,a;if(null!=t&&t<n){let s=e.shape.map(e=>0),i=e.shape.map(e=>e);i[e.shape.length-1]=t,a=B(e,s,i),n=t}else if(null!=t&&t>n){let o=e.shape.map(e=>e);o[e.shape.length-1]=t-n,a=M([e,tf(o)],e.shape.length-1),n=t}else a=e;let u=(0,ew.P)(a),l=(0,I.X)((0,X.P)(a,u),[r,n]),p=np(l),c=Math.floor(n/2)+1,d=(0,tQ.k)(p),f=(0,eX.a)(p),m=nd(d,[c,n-c],d.shape.length-1),g=nd(f,[c,n-c],f.shape.length-1),y=a.shape.slice();return y[a.shape.length-1]=c,(0,I.X)((0,X.P)(m[0],g[0]),y)}}),nm=(0,u.op)({squaredDifference_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"a","squaredDifference"),r=(0,o._1)(t,"b","squaredDifference");[n,r]=(0,v.makeTypesMatch)(n,r),(0,eN.assertAndGetBroadcastShape)(n.shape,r.shape);let a={a:n,b:r};return s.BV.runKernel(i._tC,a,{})}}),ng=(0,u.op)({squeeze_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"x","squeeze","string_or_numeric");return(0,I.X)(n,(0,h.bp)(n.shape,t).newShape)}}),ny=(0,u.op)({stack_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){let n=(0,o.sI)(e,"tensors","stack","string_or_numeric");return h.hu(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&h.hu(t<=n[0].rank,()=>"Axis must be <= rank of the tensor"),s.BV.runKernel(i.QiL,n,{axis:t})}});var nb=n(1901);let nk=(0,u.op)({stridedSlice_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a=0,u=0,l=0,p=0,c=0){let h=(0,o._1)(e,"x","stridedSlice","string_or_numeric");return s.BV.runKernel(i.jQk,{x:h},{begin:t,end:n,strides:r,beginMask:a,endMask:u,ellipsisMask:l,newAxisMask:p,shrinkAxisMask:c})}}),nN=(0,u.op)({tan_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"x","tan","float32");return s.BV.runKernel(i.sEM,{x:t})}});var nv=n(701),nx=n(7852);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nw(e,t){(0,h.Cq)(e);let n=(0,o.C)(e,t);if(1!==n.length)throw Error("tensor1d() requires values to be a flat/TypedArray");return(0,nx.H)(e,null,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nT(e,t,n){if((0,h.Cq)(e),null!=t&&2!==t.length)throw Error("tensor2d() requires shape to have two numbers");let r=(0,o.C)(e,n);if(2!==r.length&&1!==r.length)throw Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(1===r.length&&null==t)throw Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return(0,nx.H)(e,t,r,n)}var nS=n(9906);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nI(e,t,n){if((0,h.Cq)(e),null!=t&&4!==t.length)throw Error("tensor4d() requires shape to have four numbers");let r=(0,o.C)(e,n);if(4!==r.length&&1!==r.length)throw Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(1===r.length&&null==t)throw Error("tensor4d() requires shape to be provided when `values` are a flat array");return(0,nx.H)(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function n_(e,t,n){if((0,h.Cq)(e),null!=t&&5!==t.length)throw Error("tensor5d() requires shape to have five numbers");let r=(0,o.C)(e,n);if(5!==r.length&&1!==r.length)throw Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(1===r.length&&null==t)throw Error("tensor5d() requires shape to be provided when `values` are a flat array");return(0,nx.H)(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nE(e,t,n){if((0,h.Cq)(e),null!=t&&6!==t.length)throw Error("tensor6d() requires shape to have six numbers");let r=(0,o.C)(e,n);if(6!==r.length&&1!==r.length)throw Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(1===r.length&&null==t)throw Error("tensor6d() requires shape to be provided when `values` are a flat array");return t=t||r,(0,nx.H)(e,t,r,n)}let nA=(0,u.op)({topk_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=1,n=!0){let r=(0,o._1)(e,"x","topk");if(0===r.rank)throw Error("topk() expects the input to be of rank 1 or higher");let a=r.shape[r.shape.length-1];if(t<0)throw Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>a)throw Error(`'k' passed to topk() must be <= the last dimension (${a}) but got ${t}`);let[u,l]=s.BV.runKernel(i.cWu,{x:r},{k:t,sorted:n});return{values:u,indices:l}}}),nM=(0,u.op)({truncatedNormal_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0,n=1,r,a){if((0,h.Mu)(e),null!=r&&"bool"===r)throw Error("Unsupported data type $ { dtype }");let s=new tU(t,n,r,!0,a),i=(0,q.f)(e,r);for(let o=0;o<i.values.length;o++)i.values[o]=s.nextValue();return i.toTensor()}}),nD=(0,u.op)({unique_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){let n=(0,o._1)(e,"x","unique","string_or_numeric");(0,h.hu)(n.rank>0,()=>"The input tensor must be at least 1D");let[r,a]=s.BV.runKernel(i.kpP,{x:n},{axis:t});return{values:r,indices:a}}}),n$=(0,u.op)({unsortedSegmentSum_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"x","unsortedSegmentSum"),a=(0,o._1)(t,"segmentIds","unsortedSegmentSum","int32");return(0,h.hu)((0,h.GN)(n),()=>"numSegments must be of dtype int"),s.BV.runKernel(i.Qvg,{x:r,segmentIds:a},{numSegments:n})}}),nF=(0,u.op)({unstack_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){let n=(0,o._1)(e,"x","unstack","string_or_numeric");return h.hu(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`),s.BV.runKernel(i.ToN,{value:n},{axis:t})}});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nB(e,t){return to(e,t,"right")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nO(e,t=!0,n,r){return s.BV.makeVariable(e,t,n,r)}var nR=n(8333);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function nC(e){let t=(0,o._1)(e,"condition","whereAsync","bool"),n=await t.data(),r=(0,nR.Z)(t.shape,n);return e!==t&&t.dispose(),r}let nV=nC;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function nP(e,t,n){let r=(0,o._1)(e,"tensor","boolMask"),a=(0,o._1)(t,"mask","boolMask","bool"),s=null==n?0:n,i=a.rank,u=r.shape;h.hu(i>0,()=>"mask cannot be scalar"),h.k5(u.slice(s,s+i),a.shape,"mask's shape must match the first K dimensions of tensor's shape,");let l=1;for(let p=s;p<s+i;p++)l*=u[p];let c=u.slice(0,s).concat([l],u.slice(s+i)),d=(0,I.X)(r,c),f=(0,I.X)(a,[-1]),m=await nV(f),g=ng(m,[1]),y=eH(d,g,s);return e!==r&&r.dispose(),t!==a&&a.dispose(),g.dispose(),d.dispose(),f.dispose(),m.dispose(),y}let nL=nP;var nz=n(9065);let nW=(0,u.op)({movingAverage_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a=!0){let s=(0,o._1)(e,"v","movingAverage"),i=(0,o._1)(t,"x","movingAverage"),u=(0,o._1)(n,"decay","movingAverage");(0,v.assertTypesMatch)(s,i),h.hu(h.cO(s.shape,i.shape),()=>"Shape mismatch in v and x");let l=(0,eF.i)(1),p=(0,te.l)(l,u),d=(0,$.d)((0,te.l)(i,s),p);if(a){h.hu(null!=r,()=>"When using zeroDebias: true, step is required.");let f=(0,o._1)(r,"step","movingAverage");d=(0,ek.h)(d,(0,te.l)(l,(0,e$.s)(u,f)))}return(0,c.I)(s,d)}});var nU=n(3028);let nG=(0,u.op)({scatterND_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){(0,h.Mu)(n);let r=(0,o._1)(e,"indices","scatterND","int32"),a=(0,o._1)(t,"updates","scatterND");return nU.validateInput(a,r,n),s.BV.runKernel(i.xQA,{indices:r,updates:a},{shape:n})}}),nq=(0,u.op)({sparseToDense_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=0){(0,h.Mu)(n);let a=(0,o._1)(e,"sparseIndices","sparseToDense","int32"),u=(0,o._1)(t,"sparseValues","sparseToDense","string_or_numeric"),l=(0,o._1)(r,"defaultValue","sparseToDense",u.dtype);return!function(e,t,n,r){if("int32"!==e.dtype)throw Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);let a=e.rank>0?e.shape[0]:1,s=e.rank>1?e.shape[1]:1;if(n.length!==s)throw Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${s}.`);let i=t.size;if(!(0===t.rank||1===t.rank&&i===a))throw Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${a}]`);if(t.dtype!==r.dtype)throw Error("sparseValues.dtype must match defaultValues.dtype")}(a,u,n,l),s.BV.runKernel(i.D2d,{sparseIndices:a,sparseValues:u,defaultValue:l},{outputShape:n})}}),nH=(0,u.op)({gatherND_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(t,"indices","gatherND","int32"),r=(0,o._1)(e,"x","gatherND","string_or_numeric");return s.BV.runKernel(i.q1x,{params:r,indices:n})}}),nj=(0,u.op)({dropout_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r){let a=(0,o._1)(e,"x","dropout");if(h.hu("float32"===a.dtype,()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${a.dtype} tensor instead.`),h.hu(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),0===t)return e instanceof tg.es?a.clone():a;let s=/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){if(null==t)return e.shape.slice();if(h.cO(e.shape,t))return t;if(e.shape.length===t.length){let n=[];for(let r=0;r<e.shape.length;r++)null==t[r]&&null!=e.shape[r]?n.push(e.shape[r]):n.push(t[r]);return n}return t}(a,n),i=1-t,u=(0,ek.h)(eG((0,c.I)(tX(s,0,1,"float32",r),i)),i);return(0,$.d)(a,u)}});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function nK(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function nX(e,t,n){let r=1-e%2,a=new Float32Array(e);for(let s=0;s<e;++s){let i=2*Math.PI*s/(e+r-1);a[s]=t-n*Math.cos(i)}return nw(a,"float32")}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function nZ(e,t,n=1){let r=(0,o._1)(e,"predictions","inTopK"),a=(0,o._1)(t,"targets","inTopK");(0,h.hu)(r.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${r.rank}`),(0,h.hu)(r.rank-1===a.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${r.rank} and targets rank ${a.rank}`),(0,h.k5)(r.shape.slice(0,r.shape.length-1),a.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");let s=r.shape[r.shape.length-1];(0,h.hu)(n>0&&n<=s,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${s}), but got ${n}`);let i=await r.data(),u=await a.data(),[l,p]=[i.length/s,s],c=(0,h.WP)("bool",l);for(let d=0;d<l;d++){let f=d*p,m=i.subarray(f,f+p),g=[];for(let y=0;y<m.length;y++)g.push({value:m[y],index:y});g.sort((e,t)=>t.value-e.value),c[d]=0;for(let b=0;b<n;b++)if(g[b].index===u[d]){c[d]=1;break}}return e!==r&&r.dispose(),t!==a&&a.dispose(),(0,nv.X)(c,a.shape,"bool")}let nQ=nZ,nY=(0,u.op)({conv2DBackpropFilter_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,o="NHWC",u){let l=e;3===e.rank&&(l=(0,I.X)(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let p=t;3===p.rank&&(p=(0,I.X)(t,[1,t.shape[0],t.shape[1],t.shape[2]])),h.hu(4===l.rank,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${l.shape}.`),h.hu(4===p.rank,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${p.shape}.`),h.hu(4===n.length,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);let c="NHWC"===o?l.shape[3]:l.shape[1],d="NHWC"===o?p.shape[3]:p.shape[1];h.hu(c===n[2],()=>`Error in conv2dDerFilter: depth of input ${c}) must match input depth in filter (${n[2]}.`),h.hu(d===n[3],()=>`Error in conv2dDerFilter: depth of dy (${d}) must match output depth for filter (${n[3]}).`),S.m("conv2dDerFilter",a,u);let f={x:l,dy:p};return s.BV.runKernel(i.wUP,f,{strides:r,pad:a,dataFormat:o,dimRoundingMode:u,filterShape:n})}});var nJ=n(9323);let n0=(0,u.op)({fusedConv2d_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function({x:e,filter:t,strides:n,pad:r,dataFormat:a="NHWC",dilations:u=[1,1],dimRoundingMode:l,bias:p,activation:d="linear",preluActivationWeights:f,leakyreluAlpha:m}){if(d=d||"linear",!1===(0,nJ.uy)(s.BV.state.gradientDepth,d)){h.hu("NHWC"===a,()=>`Error in fused conv2d: got dataFormat of ${a} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let g=er(e,t,n,r,a,u,l);return null!=p&&(g=(0,c.I)(g,p)),(0,nJ.QH)(g,d,f,m)}let y=(0,o._1)(e,"x","conv2d","float32"),b=(0,o._1)(t,"filter","conv2d","float32"),k=y,N=!1;3===y.rank&&(N=!0,k=(0,I.X)(y,[1,y.shape[0],y.shape[1],y.shape[2]])),h.hu(4===k.rank,()=>`Error in fused conv2d: input must be rank 4, but got rank ${k.rank}.`),h.hu(4===b.rank,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${b.rank}.`),S.m("fused conv2d",r,l);let x="NHWC"===a?k.shape[3]:k.shape[1];h.hu(b.shape[2]===x,()=>`Error in conv2d: depth of input (${x}) must match input depth for filter ${b.shape[2]}.`),h.hu(S.jT(n,u),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${u}'`);let w=S.Ix(k.shape,b.shape,n,u,r,l),T;null!=p&&(T=(0,o._1)(p,"bias","fused conv2d"),[T]=(0,v.makeTypesMatch)(T,y),"NHWC"===a?eN.assertAndGetBroadcastShape(w.outShape,T.shape):(h.hu(T.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${T.shape.length}.`),h.hu(0===T.shape.length||T.shape[0]===w.outChannels||1===T.shape[0],()=>`Error in fused conv2d: bias shape (${T.shape}) is not compatible with the number of output channels (${w.outChannels})`)));let _;if(null!=f){let E=f.shape;if(h.hu(E.length<=1||3===E.length,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${E.length}.`),1===E.length)h.hu(1===E[0]||E[0]===w.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${E}) is not compatible with the number of output channels (${w.outChannels}).`);else if(3===E.length)try{eN.assertAndGetBroadcastShape(E,w.outShape)}catch(M){let A=`Error in fused conv2d: PReLU activation weights (${E}) is not compatible with the output shape of the conv2d (${w.outShape}).`;throw Error(A)}_=(0,o._1)(f,"prelu weights","fused conv2d")}let D=(e,t)=>{h.hu("NHWC"===a,()=>`Error in gradient of fused conv2D: got dataFormat of ${a} but only NHWC is currently supported.`);let[s,i,o,l]=t,p=(0,nJ.Fr)(e,o,d);h.hu(S.I0(u),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${u}'`);let c=es(i.shape,p,s,n,r),f=nY(i,p,s.shape,n,r),m=[c,f];if(null!=l){let g=(0,nJ.pf)(l,p);m.push(g)}return m},$={x:k,filter:b,bias:T,preluActivationWeights:_},F={strides:n,pad:r,dataFormat:a,dilations:u,dimRoundingMode:l,activation:d,leakyreluAlpha:m};if(null==p){let B=(0,e5.cb)((e,t,n)=>{let r=s.BV.runKernel(i._V0,$,F);return n([t,e,r]),N&&(r=(0,I.X)(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:D}});return B(k,b)}{let O=(0,e5.cb)((e,t,n,r)=>{let a=s.BV.runKernel(i._V0,$,F);return r([t,e,a,n]),N&&(a=(0,I.X)(a,[a.shape[1],a.shape[2],a.shape[3]])),{value:a,gradFunc:D}});return O(k,b,T)}}}),n1=(0,u.op)({depthwiseConv2dNativeBackpropFilter_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,o=[1,1],u){let l=e;3===e.rank&&(l=(0,I.X)(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let p=t;3===p.rank&&(p=(0,I.X)(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let c={x:l,dy:p};return s.BV.runKernel(i.sL$,c,{strides:r,pad:a,dimRoundingMode:u,dilations:o,filterShape:n})}}),n2=(0,u.op)({depthwiseConv2dNativeBackpropInput_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,o=[1,1],u){let l=t,p=!1;3===t.rank&&(p=!0,l=(0,I.X)(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let c={dy:l,filter:n},h=s.BV.runKernel(i.y7R,c,{strides:r,pad:a,dimRoundingMode:u,dilations:o,inputShape:e});return p?(0,I.X)(h,[h.shape[1],h.shape[2],h.shape[3]]):h}}),n3=(0,u.op)({fusedDepthwiseConv2d_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function({x:e,filter:t,strides:n,pad:r,dataFormat:a="NHWC",dilations:u=[1,1],dimRoundingMode:l,bias:p,activation:d="linear",preluActivationWeights:f,leakyreluAlpha:m}){if(!1===(0,nJ.uy)(s.BV.state.gradientDepth,d)){let g=eg(e,t,n,r,a,u,l);return null!=p&&(g=(0,c.I)(g,p)),(0,nJ.QH)(g,d,f,m)}let y=(0,o._1)(e,"x","depthwiseConv2d","float32"),b=(0,o._1)(t,"filter","depthwiseConv2d","float32"),k=y,N=!1;3===y.rank&&(N=!0,k=(0,I.X)(y,[1,y.shape[0],y.shape[1],y.shape[2]])),h.hu(4===k.rank,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${k.rank}.`),h.hu(4===b.rank,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${b.rank}.`),h.hu(k.shape[3]===b.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${k.shape[3]}) must match the inChannels dimension in filter ${b.shape[2]}.`),null==u&&(u=[1,1]),h.hu(S.jT(n,u),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${u}'`),S.m("fused depthwiseConv2d",r,l);let x=S.Ix(k.shape,b.shape,n,u,r,l,!0),w;null!=p&&(w=(0,o._1)(p,"bias","fused conv2d"),[w]=(0,v.makeTypesMatch)(w,y),eN.assertAndGetBroadcastShape(x.outShape,w.shape));let T;null!=f&&(T=(0,o._1)(f,"prelu weights","fused depthwiseConv2d"));let _=(e,t)=>{h.hu(S.I0(u),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${u}'`);let[a,s,i,o]=t,p=(0,nJ.Fr)(e,i,d),c=n2(s.shape,p,a,n,r,u,l),f=n1(s,p,a.shape,n,r,u,l);if(null!=o){let m=(0,nJ.pf)(w,p);return[c,f,m]}return[c,f]},E={x:k,filter:b,bias:w,preluActivationWeights:T},A={strides:n,pad:r,dataFormat:a,dilations:u,dimRoundingMode:l,activation:d,leakyreluAlpha:m};if(null==p){let M=(0,e5.cb)((e,t,n)=>{let r=s.BV.runKernel(i.luS,E,A);return n([t,e,r]),N&&(r=(0,I.X)(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:_}});return M(k,b)}{let D=(0,e5.cb)((e,t,n,r)=>{let a=s.BV.runKernel(i.luS,E,A);return r([t,e,a,n]),N&&(a=(0,I.X)(a,[a.shape[1],a.shape[2],a.shape[3]])),{value:a,gradFunc:_}});return D(k,b,w)}}}),n6=(0,u.op)({fusedMatMul_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:a,activation:u="linear",preluActivationWeights:l,leakyreluAlpha:p=.2}){if(!1===(0,nJ.uy)(s.BV.state.gradientDepth,u)){let d=(0,D.O)(e,t,n,r);return null!=a&&(d=(0,c.I)(d,a)),(0,nJ.QH)(d,u,l,p)}let f=(0,o._1)(e,"a","fused matMul"),m=(0,o._1)(t,"b","fused matMul");[f,m]=(0,v.makeTypesMatch)(f,m);let g=n?f.shape[f.rank-2]:f.shape[f.rank-1],y=r?m.shape[m.rank-1]:m.shape[m.rank-2],b=n?f.shape[f.rank-1]:f.shape[f.rank-2],k=r?m.shape[m.rank-2]:m.shape[m.rank-1],N=f.shape.slice(0,-2),x=m.shape.slice(0,-2),w=h.NA(N),T=h.NA(x);h.hu(g===y,()=>`Error in fused matMul: inner shapes (${g}) and (${y}) of Tensors with shapes ${f.shape} and ${m.shape} and transposeA=${n} and transposeB=${r} must match.`);let S=eN.assertAndGetBroadcastShape(f.shape.slice(0,-2),m.shape.slice(0,-2)),_=S.concat([b,k]),E=n?(0,I.X)(f,[w,g,b]):(0,I.X)(f,[w,b,g]),A=r?(0,I.X)(m,[T,k,y]):(0,I.X)(m,[T,y,k]),M;null!=a&&(M=(0,o._1)(a,"bias","fused matMul"),[M]=(0,v.makeTypesMatch)(M,f),eN.assertAndGetBroadcastShape(_,M.shape));let $;null!=l&&($=(0,o._1)(l,"prelu weights","fused matMul"));let F=(e,t)=>{let[s,i,o,l]=t,p=(0,nJ.Fr)((0,I.X)(e,o.shape),o,u),c,h;if(n||r?!n&&r?(c=(0,D.O)(p,i,!1,!1),h=(0,D.O)(p,s,!0,!1)):n&&!r?(c=(0,D.O)(i,p,!1,!0),h=(0,D.O)(s,p,!1,!1)):(c=(0,D.O)(i,p,!0,!0),h=(0,D.O)(p,s,!0,!0)):(c=(0,D.O)(p,i,!1,!0),h=(0,D.O)(s,p,!0,!1)),null==a)return[c,h];{let d=(0,nJ.pf)(l,p);return[c,h,d]}},B={a:E,b:A,bias:M,preluActivationWeights:$},O={transposeA:n,transposeB:r,activation:u,leakyreluAlpha:p};if(null==a){let R=(0,e5.cb)((e,t,n)=>{let r=s.BV.runKernel(i.usg,B,O);return n([e,t,r]),{value:(0,I.X)(r,_),gradFunc:F}});return R(E,A)}{let C=(0,e5.cb)((e,t,n,r)=>{let a=s.BV.runKernel(i.usg,B,O);return r([e,t,a,n]),{value:(0,I.X)(a,_),gradFunc:F}});return C(E,A,M)}}}),n4=(0,u.op)({hammingWindow_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){return nX(e,.54,.46)}}),n5=(0,u.op)({hannWindow_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){return nX(e,.5,.5)}}),n8=(0,u.op)({frame_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=!1,a=0){let s=0,i=[];for(;s+t<=e.size;)i.push(B(e,s,t)),s+=n;if(r)for(;s<e.size;){let o=s+t-e.size,u=M([B(e,s,t-o),(0,j.h)([o],a)]);i.push(u),s+=n}return 0===i.length?nT([],[0,t]):(0,I.X)(M(i),[i.length,t])}}),n7=(0,u.op)({stft_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a=n5){null==r&&(r=nK(t));let s=n8(e,t,n),i=(0,$.d)(s,a(t));return nf(i,r)}}),n9=(0,u.op)({cropAndResize_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a="bilinear",u=0){let l=(0,o._1)(e,"image","cropAndResize"),p=(0,o._1)(t,"boxes","cropAndResize","float32"),c=(0,o._1)(n,"boxInd","cropAndResize","int32"),d=p.shape[0];h.hu(4===l.rank,()=>`Error in cropAndResize: image must be rank 4,but got rank ${l.rank}.`),h.hu(2===p.rank&&4===p.shape[1],()=>`Error in cropAndResize: boxes must be have size [${d},4] but had shape ${p.shape}.`),h.hu(1===c.rank&&c.shape[0]===d,()=>`Error in cropAndResize: boxInd must be have size [${d}] but had shape ${p.shape}.`),h.hu(2===r.length,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),h.hu(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),h.hu("bilinear"===a||"nearest"===a,()=>`method must be bilinear or nearest, but was ${a}`);let f=s.BV.runKernel(i.VcC,{image:l,boxes:p,boxInd:c},{method:a,extrapolationValue:u,cropSize:r});return f}}),re=(0,u.op)({flipLeftRight_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"image","flipLeftRight","float32");h.hu(4===t.rank,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);let n=s.BV.runKernel(i.Uyb,{image:t},{});return n}}),rt=(0,u.op)({grayscaleToRGB_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,o._1)(e,"image","grayscaleToRGB"),n=t.rank-1,r=t.shape[n];h.hu(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),h.hu(1===r,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);let a=Array(t.rank);return a.fill(1,0,n),a[n]=3,eW(t,a)}}),rn=(0,u.op)({rotateWithOffset_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=0,r=.5){let a=(0,o._1)(e,"image","rotateWithOffset","float32");h.hu(4===a.rank,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${a.rank}.`);let u=s.BV.runKernel(i.b9H,{image:a},{radians:t,fillValue:n,center:r});return u}});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function rr(e,t,n,r,a,s){null==r&&(r=.5),null==a&&(a=Number.NEGATIVE_INFINITY),null==s&&(s=0);let i=e.shape[0];return n=Math.min(n,i),h.hu(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),h.hu(2===e.rank,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),h.hu(4===e.shape[1],()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),h.hu(1===t.rank,()=>"scores must be a 1D tensor"),h.hu(t.shape[0]===i,()=>`scores has incompatible shape with boxes. Expected ${i}, but was ${t.shape[0]}`),h.hu(0<=s&&s<=1,()=>`softNmsSigma must be in [0, 1], but was '${s}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:a,softNmsSigma:s}}let ra=(0,u.op)({nonMaxSuppression_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY){let u=(0,o._1)(e,"boxes","nonMaxSuppression","float32"),l=(0,o._1)(t,"scores","nonMaxSuppression","float32"),p=rr(u,l,n,r,a);n=p.maxOutputSize,r=p.iouThreshold,a=p.scoreThreshold;let c={maxOutputSize:n,iouThreshold:r,scoreThreshold:a};return s.BV.runKernel(i.uv1,{boxes:u,scores:l},c)}});var rs=n(8329);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function ri(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY){let s=(0,o._1)(e,"boxes","nonMaxSuppressionAsync"),i=(0,o._1)(t,"scores","nonMaxSuppressionAsync"),u=rr(s,i,n,r,a);n=u.maxOutputSize,r=u.iouThreshold,a=u.scoreThreshold;let l=await Promise.all([s.data(),i.data()]),p=l[0],c=l[1],{selectedIndices:h}=(0,rs.GP)(p,c,n,r,a);return s!==e&&s.dispose(),i!==t&&i.dispose(),nw(h,"int32")}let ro=(0,u.op)({nonMaxSuppressionWithScore_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,u=0){let l=(0,o._1)(e,"boxes","nonMaxSuppression"),p=(0,o._1)(t,"scores","nonMaxSuppression"),c=rr(l,p,n,r,a,u);n=c.maxOutputSize,r=c.iouThreshold,a=c.scoreThreshold,u=c.softNmsSigma;let h={maxOutputSize:n,iouThreshold:r,scoreThreshold:a,softNmsSigma:u},d=s.BV.runKernel(i.W0H,{boxes:l,scores:p},h);return{selectedIndices:d[0],selectedScores:d[1]}}});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function ru(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,s=0){let i=(0,o._1)(e,"boxes","nonMaxSuppressionAsync"),u=(0,o._1)(t,"scores","nonMaxSuppressionAsync"),l=rr(i,u,n,r,a,s);n=l.maxOutputSize,r=l.iouThreshold,a=l.scoreThreshold,s=l.softNmsSigma;let p=await Promise.all([i.data(),u.data()]),c=p[0],h=p[1],{selectedIndices:d,selectedScores:f}=(0,rs.pA)(c,h,n,r,a,s);return i!==e&&i.dispose(),u!==t&&u.dispose(),{selectedIndices:nw(d,"int32"),selectedScores:nw(f)}}let rl=(0,u.op)({nonMaxSuppressionPadded_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,u=!1){let l=(0,o._1)(e,"boxes","nonMaxSuppression"),p=(0,o._1)(t,"scores","nonMaxSuppression"),c=rr(l,p,n,r,a,null),h=c.maxOutputSize,d=c.iouThreshold,f=c.scoreThreshold,m=s.BV.runKernel(i.cye,{boxes:l,scores:p},{maxOutputSize:h,iouThreshold:d,scoreThreshold:f,padToMaxOutputSize:u});return{selectedIndices:m[0],validOutputs:m[1]}}});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ async function rp(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,s=!1){let i=(0,o._1)(e,"boxes","nonMaxSuppressionAsync"),u=(0,o._1)(t,"scores","nonMaxSuppressionAsync"),l=rr(i,u,n,r,a,null),p=l.maxOutputSize,c=l.iouThreshold,h=l.scoreThreshold,[d,f]=await Promise.all([i.data(),u.data()]),{selectedIndices:m,validOutputs:g}=(0,rs.qP)(d,f,p,c,h,s);return i!==e&&i.dispose(),u!==t&&u.dispose(),{selectedIndices:nw(m,"int32"),validOutputs:(0,eF.i)(g,"int32")}}let rc=(0,u.op)({resizeBilinear_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=!1,r=!1){let a=(0,o._1)(e,"images","resizeBilinear");h.hu(3===a.rank||4===a.rank,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${a.rank}.`),h.hu(2===t.length,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),h.hu(!1===r||!1===n,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let u=a,l=!1;3===a.rank&&(l=!0,u=(0,I.X)(a,[1,a.shape[0],a.shape[1],a.shape[2]]));let[]=t,p={images:u},c=s.BV.runKernel(i._Yw,p,{alignCorners:n,halfPixelCenters:r,size:t});return l?(0,I.X)(c,[c.shape[1],c.shape[2],c.shape[3]]):c}}),rh=(0,u.op)({resizeNearestNeighbor_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=!1,r=!1){let a=(0,o._1)(e,"images","resizeNearestNeighbor");h.hu(3===a.rank||4===a.rank,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${a.rank}.`),h.hu(2===t.length,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),h.hu("float32"===a.dtype||"int32"===a.dtype,()=>"`images` must have `int32` or `float32` as dtype"),h.hu(!1===r||!1===n,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let u=a,l=!1;3===a.rank&&(l=!0,u=(0,I.X)(a,[1,a.shape[0],a.shape[1],a.shape[2]]));let[]=t,p={images:u},c=s.BV.runKernel(i.dpD,p,{alignCorners:n,halfPixelCenters:r,size:t});return l?(0,I.X)(c,[c.shape[1],c.shape[2],c.shape[3]]):c}}),rd=(0,u.op)({threshold_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t="binary",n=!1,r=.5){let a=(0,o._1)(e,"image","threshold"),s=a.shape[0]*a.shape[1],i=(0,$.d)(nw([r]),255),u,l,p,d;if(h.hu(3===a.rank,()=>`Error in threshold: image must be rank 3,but got rank ${a.rank}.`),h.hu(3===a.shape[2]||1===a.shape[2],()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${a.shape[2]}.`),h.hu("int32"===a.dtype||"float32"===a.dtype,()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${a.dtype}.`),h.hu("otsu"===t||"binary"===t,()=>`Method must be binary or otsu, but was ${t}`),3===a.shape[2]){[u,l,p]=nd(a,[1,1,1],-1);let f=(0,$.d)(u,.2989),m=(0,$.d)(l,.587),g=(0,$.d)(p,.114);d=(0,c.I)((0,c.I)(f,m),g)}else d=e;if("otsu"===t){let y=W((0,T.p)(t5(d),"int32"),(0,nv.X)([]),256);i=function(e,t){let n=nw([-1]),r=nw([0]),a=nw([0]),s,i,o,u,l,p;for(let h=0;h<e.size-1;h++){s=B(e,0,h+1),i=B(e,h+1),l=(0,ek.h)((0,eR.S)(s),t),p=(0,ek.h)((0,eR.S)(i),t);let d=(0,eR.S)((0,$.d)(s,tZ(0,s.size)));o=(0,ek.h)(d,(0,eR.S)(s));let f=(0,j.h)(i.shape,s.size),m=(0,c.I)(tZ(0,i.size),f),g=(0,$.d)(i,m);u=(0,ek.h)((0,eR.S)(g),(0,eR.S)(i));let y=(0,te.l)(o,u),b=(0,te.l)(o,u),k=(0,$.d)(l,p);a=(0,$.d)((0,$.d)(k,y),b);let N=ej(a,r);r=ex(N,a,r),n=ex(N,nw([h]),n)}return n}(y,s)}let b=n?e1(d,i):ej(d,i),k=(0,T.p)((0,$.d)(b,255),"int32");return k}}),rf=(0,u.op)({transform_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n="nearest",r="constant",a=0,u){let l=(0,o._1)(e,"image","transform","float32"),p=(0,o._1)(t,"transforms","transform","float32");return h.hu(4===l.rank,()=>`Error in transform: image must be rank 4,but got rank ${l.rank}.`),h.hu(2===p.rank&&(p.shape[0]===l.shape[0]||1===p.shape[0])&&8===p.shape[1],()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),h.hu(null==u||2===u.length,()=>`Error in transform: outputShape must be [height, width] or null, but got ${u}.`),s.BV.runKernel(i.wx7,{image:l,transforms:p},{interpolation:n,fillMode:r,fillValue:a,outputShape:u})}}),rm=(0,u.op)({bandPart_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){(0,h.hu)(t%1==0,()=>`bandPart(): numLower must be an integer, got ${t}.`),(0,h.hu)(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`);let r=(0,o._1)(e,"a","bandPart");(0,h.hu)(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);let a=r.shape,[s,i]=r.shape.slice(-2);if(!(t<=s))throw Error(`bandPart(): numLower (${t}) must not be greater than the number of rows (${s}).`);if(!(n<=i))throw Error(`bandPart(): numUpper (${n}) must not be greater than the number of columns (${i}).`);t<0&&(t=s),n<0&&(n=i);let u=(0,I.X)(tZ(0,s,1,"int32"),[-1,1]),l=tZ(0,i,1,"int32"),p=(0,te.l)(u,l),c=tr(e1(p,(0,eF.i)(+t,"int32")),eK(p,(0,eF.i)(-n,"int32"))),d=tf([s,i],r.dtype);return(0,I.X)(ny(nF((0,I.X)(r,[-1,s,i])).map(e=>ex(c,e,d))),a)}}),rg=(0,u.op)({gramSchmidt_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t;if(Array.isArray(e)){t=!1,(0,h.hu)(null!=e&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");let n=e[0].shape[0];for(let r=1;r<e.length;++r)(0,h.hu)(e[r].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[r].shape[0]} vs. ${n})`)}else t=!0,e=nd(e,e.shape[0],0).map(e=>ng(e,[0]));(0,h.hu)(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);let a=[],i=e;for(let o=0;o<e.length;++o)a.push(s.BV.tidy(()=>{let e=i[o];if(o>0)for(let t=0;t<o;++t){let n=(0,$.d)((0,eR.S)((0,$.d)(a[t],e)),a[t]);e=(0,te.l)(e,n)}return(0,ek.h)(e,eC(e,"euclidean"))}));return t?ny(a,0):a}});var ry=n(4368);function rb(e,t=!1){return s.BV.tidy(()=>{(0,h.hu)(2===e.shape.length,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);let n=e.shape[0],r=e.shape[1],a=eU(n),i=(0,A.d)(e),o=nT([[1]],[1,1]),u=(0,A.d)(o),l=n>=r?r:n;for(let p=0;p<l;++p){let c=i,d=u,f=a;[u,i,a]=s.BV.tidy(()=>{let e=B(i,[p,p],[n-p,1]),t=eC(e),s=B(i,[p,p],[1,1]),l=ex(ej(s,0),nT([[-1]]),nT([[1]])),c=(0,te.l)(s,(0,$.d)(l,t)),h=(0,ek.h)(e,c);u=1===h.shape[0]?(0,A.d)(o):M([o,B(h,[1,0],[h.shape[0]-1,h.shape[1]])],0);let d=(0,e8.W)((0,ek.h)((0,D.O)(l,c),t)),f=B(i,[p,0],[n-p,r]),m=(0,$.d)(d,u),g=(0,nz.p)(u);if(0===p)i=(0,te.l)(f,(0,D.O)(m,(0,D.O)(g,f)));else{let y=(0,te.l)(f,(0,D.O)(m,(0,D.O)(g,f)));i=M([B(i,[0,0],[p,r]),y],0)}let b=(0,nz.p)(m),k=B(a,[0,p],[n,a.shape[1]-p]);if(0===p)a=(0,te.l)(k,(0,D.O)((0,D.O)(k,u),b));else{let N=(0,te.l)(k,(0,D.O)((0,D.O)(k,u),b));a=M([B(a,[0,0],[n,p]),N],1)}return[u,i,a]}),(0,ry.B9)([c,d,f])}return!t&&n>r&&(a=B(a,[0,0],[n,r]),i=B(i,[0,0],[r,r])),[a,i]})}let rk=(0,u.op)({qr_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=!1){if((0,h.hu)(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),2===e.rank)return rb(e,t);{let n=e.shape.slice(0,e.shape.length-2).reduce((e,t)=>e*t),r=nF((0,I.X)(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),a=[],s=[];r.forEach(e=>{let[n,r]=rb(e,t);a.push(n),s.push(r)});let i=(0,I.X)(ny(a,0),e.shape),o=(0,I.X)(ny(s,0),e.shape);return[i,o]}}});var rN=n(9876);let rv=(0,u.op)({computeWeightedLoss_:function(e,t,n=rN.I.SUM_BY_NONZERO_WEIGHTS){let r=(0,o._1)(e,"losses","computeWeightedLoss"),a=null;null!=t&&(a=(0,o._1)(t,"weights","computeWeightedLoss"));let s=null==a?r:(0,$.d)(r,a);if(n===rN.I.NONE)return s;if(n===rN.I.SUM)return(0,eR.S)(s);if(n===rN.I.MEAN){if(null==a)return td(s);{let i=r.size/a.size,u=(0,ek.h)((0,eR.S)(s),(0,eR.S)(a));return i>1?(0,ek.h)(u,(0,eF.i)(i)):u}}if(n===rN.I.SUM_BY_NONZERO_WEIGHTS){if(null==a)return(0,ek.h)((0,eR.S)(s),(0,eF.i)(r.size));{let l=(0,$.d)(a,tm(r.shape)),p=(0,T.p)((0,eR.S)(tT(l,(0,eF.i)(0))),"float32");return(0,ek.h)((0,eR.S)(s),p)}}throw Error(`Unknown reduction: ${n}`)}}),rx=(0,u.op)({absoluteDifference_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=rN.I.SUM_BY_NONZERO_WEIGHTS){let s=(0,o._1)(e,"labels","absoluteDifference"),i=(0,o._1)(t,"predictions","absoluteDifference"),u=null;null!=n&&(u=(0,o._1)(n,"weights","absoluteDifference")),(0,h.k5)(s.shape,i.shape,"Error in absoluteDifference: ");let l=(0,a.W)((0,te.l)(s,i));return rv(l,u,r)}}),rw=(0,u.op)({cosineDistance_:function(e,t,n,r,a=rN.I.SUM_BY_NONZERO_WEIGHTS){let s=(0,o._1)(e,"labels","cosineDistance"),i=(0,o._1)(t,"predictions","cosineDistance"),u=null;null!=r&&(u=(0,o._1)(r,"weights","cosineDistance")),(0,h.k5)(s.shape,i.shape,"Error in cosineDistance: ");let l=(0,eF.i)(1),p=(0,te.l)(l,(0,eR.S)((0,$.d)(s,i),n,!0));return rv(p,u,a)}}),rT=(0,u.op)({hingeLoss_:function(e,t,n,r=rN.I.SUM_BY_NONZERO_WEIGHTS){let a=(0,o._1)(e,"labels","hingeLoss"),s=(0,o._1)(t,"predictions","hingeLoss"),i=null;null!=n&&(i=(0,o._1)(n,"weights","hingeLoss")),(0,h.k5)(a.shape,s.shape,"Error in hingeLoss: ");let u=(0,eF.i)(1);a=(0,te.l)((0,$.d)((0,eF.i)(2),a),u);let l=(0,tJ.U)((0,te.l)(u,(0,$.d)(a,s)));return rv(l,i,r)}}),rS=(0,u.op)({huberLoss_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=1,s=rN.I.SUM_BY_NONZERO_WEIGHTS){let i=(0,o._1)(e,"labels","huberLoss"),u=(0,o._1)(t,"predictions","huberLoss"),l=null;null!=n&&(l=(0,o._1)(n,"weights","huberLoss")),(0,h.k5)(i.shape,u.shape,"Error in huberLoss: ");let p=(0,eF.i)(r),d=(0,a.W)((0,te.l)(u,i)),f=tb(d,p),m=(0,te.l)(d,f),g=(0,c.I)((0,$.d)((0,eF.i)(.5),(0,eO.h)(f)),(0,$.d)(p,m));return rv(g,l,s)}}),rI=(0,u.op)({logLoss_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=1e-7,a=rN.I.SUM_BY_NONZERO_WEIGHTS){let s=(0,o._1)(e,"labels","logLoss"),i=(0,o._1)(t,"predictions","logLoss"),u=null;null!=n&&(u=(0,o._1)(n,"weights","logLoss")),(0,h.k5)(s.shape,i.shape,"Error in logLoss: ");let l=(0,eF.i)(1),p=(0,eF.i)(r),d=(0,e8.W)((0,$.d)(s,e6((0,c.I)(i,p)))),f=(0,$.d)((0,te.l)(l,s),e6((0,c.I)((0,te.l)(l,i),p))),m=(0,te.l)(d,f);return rv(m,u,a)}}),r_=(0,u.op)({meanSquaredError_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r=rN.I.SUM_BY_NONZERO_WEIGHTS){let a=(0,o._1)(e,"labels","meanSquaredError"),s=(0,o._1)(t,"predictions","meanSquaredError"),i=null;null!=n&&(i=(0,o._1)(n,"weights","meanSquaredError")),(0,h.k5)(a.shape,s.shape,"Error in meanSquaredError: ");let u=nm(a,s);return rv(u,i,r)}}),rE=(0,u.op)({sigmoidCrossEntropy_:function(e,t,n,r=0,s=rN.I.SUM_BY_NONZERO_WEIGHTS){let i=(0,o._1)(e,"multiClassLabels","sigmoidCrossEntropy"),u=(0,o._1)(t,"logits","sigmoidCrossEntropy"),l=null;if(null!=n&&(l=(0,o._1)(n,"weights","sigmoidCrossEntropy")),(0,h.k5)(i.shape,u.shape,"Error in sigmoidCrossEntropy: "),r>0){let p=(0,eF.i)(r),d=(0,eF.i)(1),f=(0,eF.i)(.5);i=(0,c.I)((0,$.d)(i,(0,te.l)(d,p)),(0,$.d)(f,p))}let m=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"labels","sigmoidCrossEntropyWithLogits"),r=(0,o._1)(t,"logits","sigmoidCrossEntropyWithLogits");(0,h.k5)(n.shape,r.shape,"Error in sigmoidCrossEntropyWithLogits: ");let s=(0,tJ.U)(r),i=(0,$.d)(r,n),u=e4(eP((0,e8.W)((0,a.W)(r))));return(0,c.I)((0,te.l)(s,i),u)}(i,u);return rv(m,l,s)}}),rA=(0,u.op)({softmaxCrossEntropy_:function(e,t,n,r=0,a=rN.I.SUM_BY_NONZERO_WEIGHTS){let s=(0,o._1)(e,"onehotLabels","softmaxCrossEntropy"),i=(0,o._1)(t,"logits","softmaxCrossEntropy"),u=null;if(null!=n&&(u=(0,o._1)(n,"weights","softmaxCrossEntropy")),(0,h.k5)(s.shape,i.shape,"Error in softmaxCrossEntropy: "),r>0){let l=(0,eF.i)(r),p=(0,eF.i)(1),d=(0,eF.i)(s.shape[1]);s=(0,c.I)((0,$.d)(s,(0,te.l)(p,l)),(0,ek.h)(l,d))}let f=/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=-1){if(-1===n&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);let r=(0,e5.cb)((e,t,r)=>{let a=tn(t,[n],!0),s=(0,te.l)((0,T.p)(t,"float32"),a);r([e,s]);let i=(0,e8.W)((0,$.d)(s,e)),o=(0,eR.S)(i,[n]),u=(e,t)=>{let[r,a]=t,s=(0,eA.rv)(e.shape,[n]);return[(0,$.d)((0,I.X)(e,s),(0,te.l)((0,T.p)(r,"float32"),eP(a))),(0,$.d)((0,I.X)(e,s),(0,te.l)(eP(a),(0,T.p)(r,"float32"))),]};return{value:o,gradFunc:u}});return r(e,t)}(s,i);return rv(f,u,a)}}),rM=(0,u.op)({sparseFillEmptyRows_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r){let a=(0,o._1)(e,"indices","sparseFillEmptyRows","int32"),u=(0,o._1)(t,"values","sparseFillEmptyRows"),l=(0,o._1)(n,"denseShape","sparseFillEmptyRows","int32"),p=(0,o._1)(r,"defaultValue","sparseFillEmptyRows",u.dtype);if(2!==a.rank)throw Error(`Indices should be Tensor2D but received shape
        ${a.shape}`);if(1!==u.rank)throw Error(`Values should be Tensor1D but received shape ${u.shape}`);if(1!==l.rank)throw Error(`Dense shape should be Tensor1D but received shape ${l.shape}`);if(0!==p.rank)throw Error(`Default value should be a scalar but received shape ${p.shape}`);let c=s.BV.runKernel(i.O3z,{indices:a,values:u,denseShape:l,defaultValue:p});return{outputIndices:c[0],outputValues:c[1],emptyRowIndicator:c[2],reverseIndexMap:c[3]}}}),rD=(0,u.op)({sparseReshape_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"inputIndices","sparseReshape","int32"),a=(0,o._1)(t,"inputShape","sparseReshape","int32"),u=(0,o._1)(n,"newShape","sparseReshape","int32");if(2!==r.rank)throw Error(`Input indices should be Tensor2D but received shape
        ${r.shape}`);if(1!==a.rank)throw Error(`Input shape should be Tensor1D but received shape ${a.shape}`);if(1!==u.rank)throw Error(`New shape should be Tensor1D but received shape ${u.shape}`);let l=s.BV.runKernel(i.nhH,{inputIndices:r,inputShape:a,newShape:u});return{outputIndices:l[0],outputShape:l[1]}}}),r$=(0,u.op)({sparseSegmentMean_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"data","sparseSegmentMean"),a=(0,o._1)(t,"indices","sparseSegmentMean","int32"),u=(0,o._1)(n,"segmentIds","sparseSegmentMean","int32");if(r.rank<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.rank)throw Error(`Indices should be Tensor1D but received shape
          ${a.shape}`);if(1!==u.rank)throw Error(`Segment ids should be Tensor1D but received shape
          ${u.shape}`);return s.BV.runKernel(i.w3H,{data:r,indices:a,segmentIds:u})}}),rF=(0,u.op)({sparseSegmentSum_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let r=(0,o._1)(e,"data","sparseSegmentSum"),a=(0,o._1)(t,"indices","sparseSegmentSum","int32"),u=(0,o._1)(n,"segmentIds","sparseSegmentSum","int32");if(r.rank<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.rank)throw Error(`Indices should be Tensor1D but received shape
         ${a.shape}`);if(1!==u.rank)throw Error(`Segment ids should be Tensor1D but received shape
         ${u.shape}`);return s.BV.runKernel(i.ZjV,{data:r,indices:a,segmentIds:u})}}),rB=(0,u.op)({stringNGrams_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n,r,a,u,l,p){let c=(0,o._1)(e,"data","stringNGrams","string");if("string"!==c.dtype)throw Error("Data must be of datatype string");if(1!==c.shape.length)throw Error(`Data must be a vector, saw: ${c.shape}`);let h=(0,o._1)(t,"dataSplits","stringNGrams");if("int32"!==h.dtype)throw Error("Data splits must be of datatype int32");let d=s.BV.runKernel(i._JP,{data:c,dataSplits:h},{separator:n,nGramWidths:r,leftPad:a,rightPad:u,padWidth:l,preserveShortSequences:p});return{nGrams:d[0],nGramsSplits:d[1]}}}),rO=(0,u.op)({stringSplit_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n=!0){let r=(0,o._1)(e,"input","stringSplit","string"),a=(0,o._1)(t,"delimiter","stringSplit","string");if(1!==r.rank)throw Error(`Input should be Tensor1D but received shape ${r.shape}`);if(0!==a.rank)throw Error(`Delimiter should be a scalar but received shape ${a.shape}`);let u=s.BV.runKernel(i.s1s,{input:r,delimiter:a},{skipEmpty:n});return{indices:u[0],values:u[1],shape:u[2]}}}),rR=(0,u.op)({stringToHashBucketFast_:/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,o._1)(e,"input","stringToHashBucketFast","string");if(t<=0)throw Error("Number of buckets must be at least 1");return s.BV.runKernel(i.XkS,{input:n},{numBuckets:t})}}),rC={fft:np,ifft:nc,rfft:nf,irfft:nh},rV={hammingWindow:n4,hannWindow:n5,frame:n8,stft:n7},rP={flipLeftRight:re,grayscaleToRGB:rt,resizeNearestNeighbor:rh,resizeBilinear:rc,rotateWithOffset:rn,cropAndResize:n9,nonMaxSuppression:ra,nonMaxSuppressionAsync:ri,nonMaxSuppressionWithScore:ro,nonMaxSuppressionWithScoreAsync:ru,nonMaxSuppressionPadded:rl,nonMaxSuppressionPaddedAsync:rp,threshold:rd,transform:rf},rL={bandPart:rm,gramSchmidt:rg,qr:rk},rz={absoluteDifference:rx,computeWeightedLoss:rv,cosineDistance:rw,hingeLoss:rT,huberLoss:rS,logLoss:rI,meanSquaredError:r_,sigmoidCrossEntropy:rE,softmaxCrossEntropy:rA},rW={sparseFillEmptyRows:rM,sparseReshape:rD,sparseSegmentMean:r$,sparseSegmentSum:rF},rU={stringNGrams:rB,stringSplit:rO,stringToHashBucketFast:rR}},3453:function(e,t,n){"use strict";n.d(t,{s:function(){return u}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(2668);let u=(0,o.op)({pow_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,i._1)(e,"base","pow"),o=(0,i._1)(t,"exp","pow");[n,o]=(0,s.makeTypesMatch)(n,o);let u={a:n,b:o};return r.BV.runKernel(a.pe_,u)}})},8151:function(e,t,n){"use strict";n.d(t,{A:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({prelu_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,s._1)(e,"x","prelu"),i=(0,s._1)(t,"alpha","prelu");return r.BV.runKernel(a.o0g,{x:n,alpha:i})}})},9798:function(e,t,n){"use strict";/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function r(e,t=!1){console.log(e.toString(t))}n.d(t,{S:function(){return r}})},766:function(e,t,n){"use strict";n.d(t,{k:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({real_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"input","real");return r.BV.runKernel(a.xJR,{input:t})}})},7409:function(e,t,n){"use strict";n.d(t,{U:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({relu_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","relu");return r.BV.runKernel(a.qkr,{x:t})}})},3582:function(e,t,n){"use strict";n.d(t,{b:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({relu6_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","relu6");return r.BV.runKernel(a.SbG,{x:t})}})},4968:function(e,t,n){"use strict";n.d(t,{X:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({reshape_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,s._1)(e,"x","reshape","string_or_numeric");return r.BV.runKernel(a.HZH,{x:n},{shape:t})}})},9494:function(e,t,n){"use strict";n.d(t,{i:function(){return s}});var r=n(569),a=n(7852);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function s(e,t){if(((0,r.fU)(e)&&"string"!==t||Array.isArray(e))&&"complex64"!==t)throw Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if("string"===t&&(0,r.fU)(e)&&!(e instanceof Uint8Array))throw Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return(0,a.H)(e,[],[],t)}},3028:function(e,t,n){"use strict";n.r(t),n.d(t,{calculateShapes:function(){return i},validateInput:function(){return s},validateUpdateShape:function(){return a}});var r=n(569);function a(e,t,n){let r=t.rank>1?t.shape[t.rank-1]:1,a=t.rank>1?t.rank-1:1,s=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${a}.`;if(n.rank<a)throw Error(s+` update.rank < ${a}. `);if(e.length<r+(n.rank-a))throw Error(s+` Output shape length < ${r+(n.rank-a)}`);if(n.rank!==a+e.length-r)throw Error(s+` update.rank != ${a+e.length-r}`);for(let i=0;i<a;++i)if(n.shape[i]!==t.shape[i])throw Error(s+` updates.shape[${i}] (${n.shape[i]}) != indices.shape[${i}] (${t.shape[i]}).`);for(let o=0;o<n.rank-a;++o)if(n.shape[o+a]!==e[o+r])throw Error(s+` updates.shape[${o+a}] (${n.shape[o+a]}) != shape[${o+a}] (${e[o+a]})`)}function s(e,t,n){if(t.rank<1)throw Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if("int32"!==t.dtype)throw Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(0===n.length){if(0===t.size)throw Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(0===e.size)throw Error(`Updates specified for empty output. updates shape: ${e.shape}`)}a(n,t,e)}function i(e,t,n){let a=t.shape.length,s=a>1?t.shape[a-1]:1,i=n.length,o=1;for(let u=s;u<i;++u)o*=n[u];let l=(0,r.NA)(t.shape)/(s<1?1:s),p=[...(0,r.e3)(n.slice(0,s)),1],c=(0,r.NA)(n);return{sliceRank:s,numUpdates:l,sliceSize:o,strides:p,outputSize:c}}},625:function(e,t,n){"use strict";n.d(t,{X:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({sigmoid_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","sigmoid","float32");return r.BV.runKernel(a.a5O,{x:t})}})},3261:function(e,t,n){"use strict";n.d(t,{_:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({sqrt_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","sqrt","float32");return r.BV.runKernel(a.FKq,{x:t})}})},248:function(e,t,n){"use strict";n.d(t,{h:function(){return i}});var r=n(196),a=n(3740),s=n(2668);let i=(0,s.op)({square_:/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,a._1)(e,"x","square");return r.BV.runKernel("Square",{x:t},{})}})},1901:function(e,t,n){"use strict";n.d(t,{N:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({step_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=0){let n=(0,s._1)(e,"x","step");return r.BV.runKernel(a.h8e,{x:n},{alpha:t})}})},827:function(e,t,n){"use strict";n.d(t,{l:function(){return u}});var r=n(196),a=n(9121),s=n(747),i=n(3740),o=n(2668);let u=(0,o.op)({sub_:/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t){let n=(0,i._1)(e,"a","sub"),o=(0,i._1)(t,"b","sub");[n,o]=(0,s.makeTypesMatch)(n,o);let u={a:n,b:o};return r.BV.runKernel(a.Tr8,u)}})},5475:function(e,t,n){"use strict";n.d(t,{S:function(){return u}});var r=n(196),a=n(9121),s=n(3740),i=n(2271),o=n(2668);let u=(0,o.op)({sum_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t=null,n=!1){let o=(0,s._1)(e,"x","sum");"bool"===o.dtype&&(o=(0,i.p)(o,"int32"));let u={x:o};return r.BV.runKernel(a.GBy,u,{axis:t,keepDims:n})}})},701:function(e,t,n){"use strict";n.d(t,{X:function(){return s}});var r=n(3740),a=n(7852);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function s(e,t,n){let s=(0,r.C)(e,n);return(0,a.H)(e,t,s,n)}},9906:function(e,t,n){"use strict";n.d(t,{w:function(){return i}});var r=n(3740),a=n(569),s=n(7852);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function i(e,t,n){if((0,a.Cq)(e),null!=t&&3!==t.length)throw Error("tensor3d() requires shape to have three numbers");let i=(0,r.C)(e,n);if(3!==i.length&&1!==i.length)throw Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(1===i.length&&null==t)throw Error("tensor3d() requires shape to be provided when `values` are a flat array");return(0,s.H)(e,t,i,n)}},7852:function(e,t,n){"use strict";n.d(t,{H:function(){return i}});var r=n(196),a=n(569),s=n(3418);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function i(e,t,n,i){if(null==i&&(i=(0,a.D2)(e)),"complex64"===i)throw Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if("object"==typeof e&&"texture"in e){if("float32"!==i&&"int32"!==i)throw Error(`Creating tensor from texture only supports 'float32'|'int32' dtype, while the dtype is ${i}.`);return e.channels=e.channels||"RGBA",r.BV.backend.createTensorFromTexture(e,t||n,i)}if(!(0,a.fU)(e)&&!Array.isArray(e)&&"number"!=typeof e&&"boolean"!=typeof e&&"string"!=typeof e)throw Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(null!=t){(0,a.Mu)(t);let o=(0,a.NA)(t),u=(0,a.NA)(n);(0,a.hu)(o===u,()=>`Based on the provided shape, [${t}], the tensor should have ${o} values but has ${u}`);for(let l=0;l<n.length;++l){let p=n[l],c=l!==n.length-1||p!==(0,a.NA)(t.slice(l));(0,a.hu)(n[l]===t[l]||!c,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return(0,a.fU)(e)||Array.isArray(e)||(e=[e]),t=t||n,e="string"!==i?(0,s.toTypedArray)(e,i):(0,a.xH)(e,[],!0),r.BV.makeTensor(e,t,i)}},9065:function(e,t,n){"use strict";n.d(t,{p:function(){return d}});var r=n(196),a=n(4368),s=n(9121),i=n(3740),o=n(569),u=n(1661),l=n(4386),p=n(7370),c=n(2668),h=n(766);let d=(0,c.op)({transpose_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e,t,n){let c=(0,i._1)(e,"x","transpose");if(null==t&&(t=c.shape.map((e,t)=>t).reverse()),o.hu(c.rank===t.length,()=>`Error in transpose: rank of input ${c.rank} must match length of perm ${t}.`),t.forEach(e=>{o.hu(e>=0&&e<c.rank,()=>`All entries in 'perm' must be between 0 and ${c.rank-1} but got ${t}`)}),c.rank<=1)return c.clone();let d={perm:t};return"complex64"===c.dtype?(0,a.lu)(()=>{let e=(0,h.k)(c),t=(0,l.a)(c);return e=r.BV.runKernel(s.G3Y,{x:e},d),t=r.BV.runKernel(s.G3Y,{x:t},d),n&&(t=(0,p.W)(t)),(0,u.P)(e,t)}):r.BV.runKernel(s.G3Y,{x:c},d)}})},6577:function(e,t,n){"use strict";n.d(t,{P:function(){return o}});var r=n(196),a=n(9121),s=n(3740),i=n(2668);let o=(0,i.op)({zerosLike_:/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function(e){let t=(0,s._1)(e,"x","zerosLike");return r.BV.runKernel(a.RuY,{x:t})}})},974:function(e,t,n){"use strict";n.d(t,{es:function(){return m},YD:function(){return l},_w:function(){return g},FZ:function(){return f},Vp:function(){return d},Vi:function(){return h}});var r=n(5938),a=n(569);function s(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(7))} + ${parseFloat(e[1].toFixed(7))}j`:(0,a.HD)(e)?`'${e}'`:"bool"===n?i(e):parseFloat(e.toFixed(7)).toString(),(0,a.oj)(r,t)}function i(e){return 0===e?"false":"true"}function o(e){let t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}var u=n(3418);/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ class l{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=a.NA(e),null!=n){let r=n.length;a.hu(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if("complex64"===t)throw Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=n||a.rQ(t,this.size),this.strides=(0,a.e3)(e)}set(e,...t){0===t.length&&(t=[0]),a.hu(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);let n=this.locToIndex(t);this.values[n]=e}get(...e){0===e.length&&(e=[0]);let t=0;for(let n of e){if(n<0||n>=this.shape[t]){let r=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw Error(r)}t++}let a=e[e.length-1];for(let s=0;s<e.length-1;++s)a+=this.strides[s]*e[s];return this.values[a]}locToIndex(e){if(0===this.rank)return 0;if(1===this.rank)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(0===this.rank)return[];if(1===this.rank)return[e];let t=Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return p().makeTensor(this.values,this.shape,this.dtype)}}let p=null,c=null;function h(e){p=e}function d(e){c=e}function f(e){}class m{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||"float32",this.size=a.NA(e),this.strides=(0,a.e3)(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){let e=await this.data();return c.buffer(this.shape,this.dtype,e)}bufferSync(){return c.buffer(this.shape,this.dtype,this.dataSync())}async array(){let e=await this.data();return(0,a.GX)(this.shape,e,"complex64"===this.dtype)}arraySync(){return(0,a.GX)(this.shape,this.dataSync(),"complex64"===this.dtype)}async data(){this.throwIfDisposed();let e=p().read(this.dataId);if("string"===this.dtype){let t=await e;try{return t.map(e=>u.decodeString(e))}catch(n){throw Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),p().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();let e=p().readSync(this.dataId);if("string"===this.dtype)try{return e.map(e=>u.decodeString(e))}catch(t){throw Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();let e=await p().read(this.dataId);return"string"===this.dtype?e:new Uint8Array(e.buffer)}dispose(){!this.isDisposed&&(p().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw Error("Tensor is disposed.")}print(e=!1){return c.print(this,e)}clone(){return this.throwIfDisposed(),c.clone(this)}toString(e=!1){let t=this.dataSync();return function(e,t,n,r){let u=(0,a.e3)(t),l=function(e,t,n,r){let i=(0,a.NA)(t),u=r[r.length-1],l=Array(u).fill(0),p=t.length,c="complex64"===n?o(e):e;if(p>1)for(let h=0;h<i/u;h++){let d=h*u;for(let f=0;f<u;f++)l[f]=Math.max(l[f],s(c[d+f],0,n).length)}return l}(e,t,n,u),p=t.length,c=function e(t,n,r,a,u,l=!0){let p="complex64"===r?2:1,c=n[0],h=n.length;if(0===h){if("complex64"===r){let d=o(t);return[s(d[0],0,r)]}return"bool"===r?[i(t[0])]:[t[0].toString()]}if(1===h){if(c>20){let f=Array.from(t.slice(0,3*p)),m=Array.from(t.slice((c-3)*p,c*p));return"complex64"===r&&(f=o(f),m=o(m)),["["+f.map((e,t)=>s(e,u[t],r)).join(", ")+", ..., "+m.map((e,t)=>s(e,u[c-3+t],r)).join(", ")+"]"]}let g="complex64"===r?o(t):Array.from(t);return["["+g.map((e,t)=>s(e,u[t],r)).join(", ")+"]"]}let y=n.slice(1),b=a.slice(1),k=a[0]*p,N=[];if(c>20){for(let v=0;v<3;v++){let x=v*k,w=x+k;N.push(...e(t.slice(x,w),y,r,b,u,!1))}N.push("...");for(let T=c-3;T<c;T++){let S=T*k,I=S+k;N.push(...e(t.slice(S,I),y,r,b,u,T===c-1))}}else for(let _=0;_<c;_++){let E=_*k,A=E+k;N.push(...e(t.slice(E,A),y,r,b,u,_===c-1))}let M=2===h?",":"";N[0]="["+N[0]+M;for(let D=1;D<N.length-1;D++)N[D]=" "+N[D]+M;let $=",\n";for(let F=2;F<h;F++)$+="\n";return N[N.length-1]=" "+N[N.length-1]+"]"+(l?"":$),N}(e,t,n,u,l),h=["Tensor"];return r&&(h.push(`  dtype: ${n}`),h.push(`  rank: ${p}`),h.push(`  shape: [${t}]`),h.push("  values:")),h.push(c.map(e=>"    "+e).join("\n")),h.join("\n")}(t,this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),c.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),p().makeVariable(this,e,t,n)}}Object.defineProperty(m,Symbol.hasInstance,{value:e=>!!e&&null!=e.data&&null!=e.dataSync&&null!=e.throwIfDisposed}),(0,r.R)("Tensor",()=>m);class g extends m{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!a.cO(e.shape,this.shape))throw Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);p().disposeTensor(this),this.dataId=e.dataId,p().incRef(this,null)}dispose(){p().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(g,Symbol.hasInstance,{value:e=>e instanceof m&&null!=e.assign&&e.assign instanceof Function})},747:function(e,t,n){"use strict";n.r(t),n.d(t,{assertTypesMatch:function(){return o},getTensorsInContainer:function(){return l},isTensorInList:function(){return u},makeTypesMatch:function(){return i}});var r=n(974),a=n(1221),s=n(569);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function i(e,t){if(e.dtype===t.dtype)return[e,t];let n=(0,a.x8)(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function o(e,t){(0,s.hu)(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function u(e,t){return t.some(t=>t.id===e.id)}function l(e){let t=[],n=new Set;return function e(t,n,a){var s;if(null!=t){if(t instanceof r.es){n.push(t);return}if(s=t,Array.isArray(s)||"object"==typeof s)for(let i in t){let o=t[i];a.has(o)||(a.add(o),e(o,n,a))}}}(e,t,n),t}},3740:function(e,t,n){"use strict";n.d(t,{C:function(){return u},_1:function(){return p},sI:function(){return c}});var r=n(196),a=n(2885),s=n(974),i=n(569),o=n(3418);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function u(e,t){let n=e;if((0,i.fU)(e))return"string"===t?[]:[e.length];if("object"==typeof e&&"texture"in e){let r=e.channels||"RGBA";return[e.height,e.width*r.length]}if(!Array.isArray(e))return[];let s=[];for(;Array.isArray(n)||(0,i.fU)(n)&&"string"!==t;)s.push(n.length),n=n[0];return Array.isArray(e)&&(0,a.OB)().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&function e(t,n,r){if(r=r||[],!Array.isArray(t)&&!(0,i.fU)(t)){(0,i.hu)(0===n.length,()=>`Element arr[${r.join("][")}] is a primitive, but should be an array/TypedArray of ${n[0]} elements`);return}(0,i.hu)(n.length>0,()=>`Element arr[${r.join("][")}] should be a primitive, but is an array of ${t.length} elements`),(0,i.hu)(t.length===n[0],()=>`Element arr[${r.join("][")}] should have ${n[0]} elements, but has ${t.length} elements`);let a=n.slice(1);for(let s=0;s<t.length;++s)e(t[s],a,r.concat(s))}(e,s,[]),s}function l(e,t,n,r){if("string_or_numeric"!==e){if(null==e)throw Error("Expected dtype cannot be null.");if("numeric"!==e&&e!==t||"numeric"===e&&"string"===t)throw Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function p(e,t,n,a="numeric"){if(e instanceof s.es)return l(a,e.dtype,t,n),e;let p=(0,i.D2)(e);if("string"!==p&&["bool","int32","float32"].indexOf(a)>=0&&(p=a),l(a,p,t,n),null==e||!(0,i.fU)(e)&&!Array.isArray(e)&&"number"!=typeof e&&"boolean"!=typeof e&&"string"!=typeof e){let c=null==e?"null":e.constructor.name;throw Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${c}'`)}let h=u(e,p);(0,i.fU)(e)||Array.isArray(e)||(e=[e]);let d="string"!==p?(0,o.toTypedArray)(e,p):(0,i.xH)(e,[],!0);return r.BV.makeTensor(d,h,p)}function c(e,t,n,r="numeric"){if(!Array.isArray(e))throw Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,a)=>p(e,`${t}[${a}]`,n,r))}},1221:function(e,t,n){"use strict";var r,a,s,i,o,u,l,p,c,h;n.d(t,{x8:function(){return f},yw:function(){return r},z4:function(){return m}}),(u=r||(r={})).R0="R0",u.R1="R1",u.R2="R2",u.R3="R3",u.R4="R4",u.R5="R5",u.R6="R6",(l=a||(a={})).float32="float32",l.int32="int32",l.bool="int32",l.complex64="complex64",(p=s||(s={})).float32="float32",p.int32="int32",p.bool="bool",p.complex64="complex64",(c=i||(i={})).float32="float32",c.int32="float32",c.bool="float32",c.complex64="complex64",(h=o||(o={})).float32="complex64",h.int32="complex64",h.bool="complex64",h.complex64="complex64";let d={float32:i,int32:a,bool:s,complex64:o};function f(e,t){if("string"===e||"string"===t){if("string"===e&&"string"===t)return"string";throw Error(`Can not upcast ${e} with ${t}`)}return d[e][t]}function m(e){return f(e,"int32")}},3418:function(e,t,n){"use strict";n.r(t),n.d(t,{arraysEqual:function(){return a.cO},assert:function(){return a.hu},assertNonNegativeIntegerDimensions:function(){return a.Mu},assertNonNull:function(){return a.Cq},assertShapesMatch:function(){return a.k5},bytesFromStringArray:function(){return a.Ub},bytesPerElement:function(){return a.bT},checkConversionForErrors:function(){return a.D5},clamp:function(){return a.uZ},computeStrides:function(){return a.e3},createScalarValue:function(){return N},createShuffledIndices:function(){return a.U$},decodeString:function(){return S},distSquared:function(){return a.E7},encodeString:function(){return T},fetch:function(){return w},fingerPrint64:function(){return k},flatten:function(){return a.xH},getArrayFromDType:function(){return a.rQ},getTypedArrayFromDType:function(){return a.WP},hasEncodingLoss:function(){return a.QB},hexToLong:function(){return u},indexToLoc:function(){return a.NE},inferDtype:function(){return a.D2},inferFromImplicitShape:function(){return a.JZ},isBoolean:function(){return a.jn},isFunction:function(){return a.mf},isInt:function(){return a.GN},isNumber:function(){return a.hj},isPromise:function(){return a.tI},isScalarShape:function(){return a.N9},isString:function(){return a.HD},isTypedArray:function(){return a.fU},isValidDtype:function(){return a.LP},locToIndex:function(){return a.qy},makeOnesTypedArray:function(){return a.p8},makeZerosNestedTypedArray:function(){return a.l6},makeZerosTypedArray:function(){return a.wT},nearestDivisor:function(){return a.jP},nearestLargerEven:function(){return a.nY},now:function(){return x},parseAxisParam:function(){return a.EC},randUniform:function(){return a.bj},repeatedTry:function(){return a.WD},rightPad:function(){return a.oj},shuffle:function(){return a.TV},shuffleCombo:function(){return a.d7},sizeFromShape:function(){return a.NA},sizeToSquarishShape:function(){return a.YP},squeezeShape:function(){return a.bp},sum:function(){return a.Sm},swap:function(){return a.LF},tanh:function(){return a.AE},toNestedArray:function(){return a.GX},toTypedArray:function(){return v}});var r=n(2885),a=n(569),s=n(3720),i=n.n(s);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ let o=i()||s;function u(e){return o.fromString(e,!0,16)}let l=u("c3a5c85c97cb3127"),p=u("b492b66fbe98f273"),c=u("9ae16a3b2f90404f");function h(e){return e.xor(e.shru(47))}function d(e,t,n){let r=e.slice(t,t+n);return o.fromBytes(Array.from(r),!0,!0)}function f(e,t){return d(e,t,8)}function m(e,t){return d(e,t,4)}function g(e,t){return 0===t?e:e.shru(t).or(e.shl(64-t))}function y(e,t,n=u("9ddfea08eb382d69")){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let a=t.xor(r).mul(n);return(a=a.xor(a.shru(47))).mul(n)}function b(e,t,n,r){return function(e,t,n,r,a,s){a=a.add(e),s=g(s.add(a).add(r),21);let i=a;return a=(a=a.add(t)).add(n),s=s.add(g(a,44)),[a.add(r),s.add(i)]}(f(e,t),f(e,t+8),f(e,t+16),f(e,t+24),n,r)}function k(e,t=e.length){let n=o.fromNumber(81,!0);if(t<=32)return t<=16?function(e,t=e.length){if(t>=8){let n=c.add(2*t),r=f(e,0).add(c),a=f(e,t-8),s=g(a,37).mul(n).add(r),i=g(r,25).add(a).mul(n);return y(s,i,n)}if(t>=4){let o=c.add(2*t),u=m(e,0);return y(u.shl(3).add(t),m(e,t-4),o)}if(t>0){let p=e[0],d=e[t>>1],b=e[t-1];return h(c.mul(p+(d<<8)).xor(l.mul(t+(b<<2)))).mul(c)}return c}(e,t):function(e,t=e.length){let n=c.add(2*t),r=f(e,0).mul(p),a=f(e,8),s=f(e,t-8).mul(n),i=f(e,t-16).mul(c);return y(g(r.add(a),43).add(g(s,30)).add(i),r.add(g(a.add(c),18)).add(s),n)}(e,t);if(t<=64)return function(e,t=e.length){let n=c.add(2*t),r=f(e,0).mul(c),a=f(e,8),s=f(e,t-8).mul(n),i=f(e,t-16).mul(c),o=g(r.add(a),43).add(g(s,30)).add(i),u=y(o,r.add(g(a.add(c),18)).add(s),n),l=f(e,16).mul(n),p=f(e,24),h=o.add(f(e,t-32)).mul(n),d=u.add(f(e,t-24)).mul(n);return y(g(l.add(p),43).add(g(h,30)).add(d),l.add(g(p.add(r),18)).add(h),n)}(e,t);let r=n,a=n.mul(p).add(113),s=h(a.mul(c).add(113)).mul(c),i=[o.UZERO,o.UZERO],u=[o.UZERO,o.UZERO];r=r.mul(c).add(f(e,0));let d=0,k=(t-1>>6)*64;do r=g(r.add(a).add(i[0]).add(f(e,d+8)),37).mul(p),a=g(a.add(i[1]).add(f(e,d+48)),42).mul(p),r=r.xor(u[1]),a=a.add(i[0]).add(f(e,d+40)),s=g(s.add(u[0]),33).mul(p),i=b(e,d,i[1].mul(p),r.add(u[0])),u=b(e,d+32,s.add(u[1]),a.add(f(e,d+16))),[s,r]=[r,s],d+=64;while(d!==k);let N=p.add(s.and(255).shl(1));return d=k+(t-1&63)-63,u[0]=u[0].add(t-1&63),i[0]=i[0].add(u[0]),u[0]=u[0].add(i[0]),r=g(r.add(a).add(i[0]).add(f(e,d+8)),37).mul(N),a=g(a.add(i[1]).add(f(e,d+48)),42).mul(N),r=r.xor(u[1].mul(9)),a=a.add(i[0].mul(9).add(f(e,d+40))),s=g(s.add(u[0]),33).mul(N),i=b(e,d,i[1].mul(N),r.add(u[0])),u=b(e,d+32,s.add(u[1]),a.add(f(e,d+16))),[s,r]=[r,s],y(y(i[0],u[0],N).add(h(a).mul(l)).add(s),y(i[1],u[1],N).add(r),N)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function N(e,t){return"string"===t?T(e):v([e],t)}function v(e,t){var n,s;if("string"===t)throw Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=a.xH(e)),(0,r.OB)().getBool("DEBUG")&&a.D5(e,t),(n=e)instanceof Float32Array&&"float32"===t||n instanceof Int32Array&&"int32"===t||n instanceof Uint8Array&&"bool"===t)return e;if(null==t||"float32"===t||"complex64"===t)return new Float32Array(e);if("int32"===t)return new Int32Array(e);if("bool"===t){let i=new Uint8Array(e.length);for(let o=0;o<i.length;++o)0!==Math.round(e[o])&&(i[o]=1);return i}throw Error(`Unknown data type ${t}`)}function x(){return(0,r.OB)().platform.now()}function w(e,t){return(0,r.OB)().platform.fetch(e,t)}function T(e,t="utf-8"){return t=t||"utf-8",(0,r.OB)().platform.encode(e,t)}function S(e,t="utf-8"){return t=t||"utf-8",(0,r.OB)().platform.decode(e,t)}},569:function(e,t,n){"use strict";/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ function r(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,o(e,--t,n)}function a(e,t){if(e.length!==t.length)throw Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,r=0;for(;n>0;)r=Math.random()*n|0,o(e,--n,r),o(t,n,r)}function s(e,t,n){return Math.max(e,Math.min(t,n))}function i(e){return e%2==0?e:e+1}function o(e,t,n){let r=e[t];e[t]=e[n],e[n]=r}function u(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function l(e,t){let n=Math.random();return t*n+(1-n)*e}function p(e,t){let n=0;for(let r=0;r<e.length;r++){let a=Number(e[r])-Number(t[r]);n+=a*a}return n}function c(e,t){if(!e)throw Error("string"==typeof t?t:t())}function h(e,t,n=""){c(g(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function d(e){c(null!=e,()=>"The input to the tensor constructor must be a non-null value.")}function f(e){if(0===e.length)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function m(e){return 0===e.length}function g(e,t){if(e===t)return!0;if(null==e||null==t||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function y(e){return e%1==0}function b(e){if(null!=Math.tanh)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return -1;{let t=Math.exp(2*e);return(t-1)/(t+1)}}function k(e){let t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function N(e){let t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return r(t),t}function v(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function x(e,t=e=>0,n,r){return new Promise((a,s)=>{let i=0,o=()=>{if(e()){a();return}i++;let u=t(i);if(null!=n&&i>=n){s();return}null!=r?r(o,u):setTimeout(o,u)};o()})}function w(e,t){let n=1,r=-1;for(let a=0;a<e.length;++a)if(e[a]>=0)n*=e[a];else if(-1===e[a]){if(-1!==r)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${a}`);r=a}else if(e[a]<0)throw Error(`Shapes can not be < 0. Found ${e[a]} at dim ${a}`);if(-1===r){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(0===n)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!=0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);let s=e.slice();return s[r]=t/n,s}function T(e,t){let n=t.length;return c((e=null==e?t.map((e,t)=>t):[].concat(e)).every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),c(e.every(e=>y(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function S(e,t){let n=[],r=[],a=null!=t&&Array.isArray(t)&&0===t.length,s=null==t||a?null:T(t,e).sort(),i=0;for(let o=0;o<e.length;++o){if(null!=s){if(s[i]===o&&1!==e[o])throw Error(`Can't squeeze axis ${o} since its dim '${e[o]}' is not 1`);(null==s[i]||s[i]>o)&&1===e[o]&&(n.push(e[o]),r.push(o)),s[i]<=o&&i++}1!==e[o]&&(n.push(e[o]),r.push(o))}return{newShape:n,keptDims:r}}function I(e,t){let n=null;if(null==e||"float32"===e)n=new Float32Array(t);else if("int32"===e)n=new Int32Array(t);else if("bool"===e)n=new Uint8Array(t);else throw Error(`Unknown data type ${e}`);return n}function _(e,t){let n=null;if(null==e||"float32"===e)n=new Float32Array(t);else if("int32"===e)n=new Int32Array(t);else if("bool"===e)n=new Uint8Array(t);else if("string"===e)n=Array(t);else throw Error(`Unknown data type ${e}`);return n}function E(e,t){for(let n=0;n<e.length;n++){let r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function A(e){return"bool"===e||"complex64"===e||"float32"===e||"int32"===e||"string"===e}function M(e,t){return"complex64"!==t&&("float32"!==t||"complex64"===e)&&("int32"!==t||"float32"===e||"complex64"===e)&&("bool"!==t||"bool"!==e)}function D(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}function $(e){if("float32"===e||"int32"===e)return 4;if("complex64"===e)return 8;if("bool"===e)return 1;throw Error(`Unknown dtype ${e}`)}function F(e){if(null==e)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function B(e){return"string"==typeof e||e instanceof String}function O(e){return"boolean"==typeof e}function R(e){return"number"==typeof e}function C(e){return!!(e&&e.constructor&&e.call&&e.apply)}function V(e,t){for(let n=t;n<e;++n)if(e%n==0)return n;return e}function P(e){let t=e.length;if(t<2)return[];let n=Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function L(e,t,n=!1){if(0===e.length)return t[0];let r=e.reduce((e,t)=>e*t)*(n?2:1);if(0===r)return[];if(r!==t.length)throw Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return function e(t,n,r,a=!1){let s=[];if(1===n.length){let i=n[0]*(a?2:1);for(let o=0;o<i;o++)s[o]=r[t+o]}else{let u=n[0],l=n.slice(1),p=l.reduce((e,t)=>e*t)*(a?2:1);for(let c=0;c<u;c++)s[c]=e(t+c*p,l,r,a)}return s}(0,e,t,n)}function z(e,t){let n=W(e,t);for(let r=0;r<n.length;r++)n[r]=1;return n}function W(e,t){if(null==t||"float32"===t||"complex64"===t)return new Float32Array(e);if("int32"===t)return new Int32Array(e);if("bool"===t)return new Uint8Array(e);throw Error(`Unknown data type ${t}`)}function U(e,t){let n=e.reduce((e,t)=>e*t,1);if(null==t||"float32"===t)return L(e,new Float32Array(n));if("int32"===t)return L(e,new Int32Array(n));if("bool"===t)return L(e,new Uint8Array(n));throw Error(`Unknown data type ${t}`)}function G(e){e.forEach(t=>{c(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function q(e,t,n){if(0===t)return 0;if(1===t)return e[0];let r=e[e.length-1];for(let a=0;a<e.length-1;++a)r+=n[a]*e[a];return r}function H(e,t,n){if(0===t)return[];if(1===t)return[e];let r=Array(t);for(let a=0;a<r.length-1;++a)r[a]=Math.floor(e/n[a]),e-=r[a]*n[a];return r[r.length-1]=e,r}function j(e){return e&&e.then&&"function"==typeof e.then}n.d(t,{AE:function(){return b},Cq:function(){return d},D2:function(){return function e(t){if(Array.isArray(t))return e(t[0]);if(t instanceof Float32Array);else if(t instanceof Int32Array||t instanceof Uint8Array||t instanceof Uint8ClampedArray)return"int32";else if(R(t));else if(B(t))return"string";else if(O(t))return"bool";return"float32"}},D5:function(){return E},E7:function(){return p},EC:function(){return T},GN:function(){return y},GX:function(){return L},HD:function(){return B},JZ:function(){return w},LF:function(){return o},LP:function(){return A},Mu:function(){return G},N9:function(){return m},NA:function(){return f},NE:function(){return H},QB:function(){return M},Sm:function(){return u},TV:function(){return r},U$:function(){return N},Ub:function(){return F},WD:function(){return x},WP:function(){return I},YP:function(){return k},bT:function(){return $},bj:function(){return l},bp:function(){return S},cO:function(){return g},d7:function(){return a},e3:function(){return P},fU:function(){return D},hj:function(){return R},hu:function(){return c},jP:function(){return V},jn:function(){return O},k5:function(){return h},l6:function(){return U},mf:function(){return C},nY:function(){return i},oj:function(){return v},p8:function(){return z},qy:function(){return q},rQ:function(){return _},tI:function(){return j},uZ:function(){return s},wT:function(){return W},xH:function(){return function e(t,n=[],r=!1){if(null==n&&(n=[]),Array.isArray(t)||D(t)&&!r)for(let a=0;a<t.length;++a)e(t[a],n,r);else n.push(t);return n}}})},3720:function(e){e.exports=r;var t=null;try{t=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch(n){}function r(e,t,n){this.low=0|e,this.high=0|t,this.unsigned=!!n}function a(e){return!0===(e&&e.__isLong__)}r.prototype.__isLong__,Object.defineProperty(r.prototype,"__isLong__",{value:!0}),r.isLong=a;var s={},i={};function o(e,t){var n,r,a;return t?(e>>>=0,(a=0<=e&&e<256)&&(r=i[e]))?r:(n=l(e,(0|e)<0?-1:0,!0),a&&(i[e]=n),n):(e|=0,(a=-128<=e&&e<128)&&(r=s[e]))?r:(n=l(e,e<0?-1:0,!1),a&&(s[e]=n),n)}function u(e,t){if(isNaN(e))return t?b:y;if(t){if(e<0)return b;if(e>=f)return w}else{if(e<=-m)return T;if(e+1>=m)return x}return e<0?u(-e,t).neg():l(e%d|0,e/d|0,t)}function l(e,t,n){return new r(e,t,n)}r.fromInt=o,r.fromNumber=u,r.fromBits=l;var p=Math.pow;function c(e,t,n){if(0===e.length)throw Error("empty string");if("NaN"===e||"Infinity"===e||"+Infinity"===e||"-Infinity"===e)return y;if("number"==typeof t?(n=t,t=!1):t=!!t,(n=n||10)<2||36<n)throw RangeError("radix");if((r=e.indexOf("-"))>0)throw Error("interior hyphen");if(0===r)return c(e.substring(1),t,n).neg();for(var r,a=u(p(n,8)),s=y,i=0;i<e.length;i+=8){var o=Math.min(8,e.length-i),l=parseInt(e.substring(i,i+o),n);if(o<8){var h=u(p(n,o));s=s.mul(h).add(u(l))}else s=(s=s.mul(a)).add(u(l))}return s.unsigned=t,s}function h(e,t){return"number"==typeof e?u(e,t):"string"==typeof e?c(e,t):l(e.low,e.high,"boolean"==typeof t?t:e.unsigned)}r.fromString=c,r.fromValue=h;var d=4294967296,f=d*d,m=f/2,g=o(16777216),y=o(0);r.ZERO=y;var b=o(0,!0);r.UZERO=b;var k=o(1);r.ONE=k;var N=o(1,!0);r.UONE=N;var v=o(-1);r.NEG_ONE=v;var x=l(-1,2147483647,!1);r.MAX_VALUE=x;var w=l(-1,-1,!0);r.MAX_UNSIGNED_VALUE=w;var T=l(0,-2147483648,!1);r.MIN_VALUE=T;var S=r.prototype;S.toInt=function(){return this.unsigned?this.low>>>0:this.low},S.toNumber=function(){return this.unsigned?(this.high>>>0)*d+(this.low>>>0):this.high*d+(this.low>>>0)},S.toString=function(e){if((e=e||10)<2||36<e)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative()){if(!this.eq(T))return"-"+this.neg().toString(e);var t=u(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}for(var a=u(p(e,6),this.unsigned),s=this,i="";;){var o=s.div(a),l=(s.sub(o.mul(a)).toInt()>>>0).toString(e);if((s=o).isZero())return l+i;for(;l.length<6;)l="0"+l;i=""+l+i}},S.getHighBits=function(){return this.high},S.getHighBitsUnsigned=function(){return this.high>>>0},S.getLowBits=function(){return this.low},S.getLowBitsUnsigned=function(){return this.low>>>0},S.getNumBitsAbs=function(){if(this.isNegative())return this.eq(T)?64:this.neg().getNumBitsAbs();for(var e=0!=this.high?this.high:this.low,t=31;t>0&&(e&1<<t)==0;t--);return 0!=this.high?t+33:t+1},S.isZero=function(){return 0===this.high&&0===this.low},S.eqz=S.isZero,S.isNegative=function(){return!this.unsigned&&this.high<0},S.isPositive=function(){return this.unsigned||this.high>=0},S.isOdd=function(){return(1&this.low)==1},S.isEven=function(){return(1&this.low)==0},S.equals=function(e){return a(e)||(e=h(e)),(this.unsigned===e.unsigned||this.high>>>31!=1||e.high>>>31!=1)&&this.high===e.high&&this.low===e.low},S.eq=S.equals,S.notEquals=function(e){return!this.eq(e)},S.neq=S.notEquals,S.ne=S.notEquals,S.lessThan=function(e){return 0>this.comp(e)},S.lt=S.lessThan,S.lessThanOrEqual=function(e){return 0>=this.comp(e)},S.lte=S.lessThanOrEqual,S.le=S.lessThanOrEqual,S.greaterThan=function(e){return this.comp(e)>0},S.gt=S.greaterThan,S.greaterThanOrEqual=function(e){return this.comp(e)>=0},S.gte=S.greaterThanOrEqual,S.ge=S.greaterThanOrEqual,S.compare=function(e){if(a(e)||(e=h(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},S.comp=S.compare,S.negate=function(){return!this.unsigned&&this.eq(T)?T:this.not().add(k)},S.neg=S.negate,S.add=function(e){a(e)||(e=h(e));var t=this.high>>>16,n=65535&this.high,r=this.low>>>16,s=65535&this.low,i=e.high>>>16,o=65535&e.high,u=e.low>>>16,p=65535&e.low,c=0,d=0,f=0,m=0;return m+=s+p,f+=m>>>16,m&=65535,f+=r+u,d+=f>>>16,f&=65535,d+=n+o,c+=d>>>16,d&=65535,c+=t+i,l(f<<16|m,(c&=65535)<<16|d,this.unsigned)},S.subtract=function(e){return a(e)||(e=h(e)),this.add(e.neg())},S.sub=S.subtract,S.multiply=function(e){if(this.isZero())return y;if(a(e)||(e=h(e)),t)return l(t.mul(this.low,this.high,e.low,e.high),t.get_high(),this.unsigned);if(e.isZero())return y;if(this.eq(T))return e.isOdd()?T:y;if(e.eq(T))return this.isOdd()?T:y;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(g)&&e.lt(g))return u(this.toNumber()*e.toNumber(),this.unsigned);var n=this.high>>>16,r=65535&this.high,s=this.low>>>16,i=65535&this.low,o=e.high>>>16,p=65535&e.high,c=e.low>>>16,d=65535&e.low,f=0,m=0,b=0,k=0;return k+=i*d,b+=k>>>16,k&=65535,b+=s*d,m+=b>>>16,b&=65535,b+=i*c,m+=b>>>16,b&=65535,m+=r*d,f+=m>>>16,m&=65535,m+=s*c,f+=m>>>16,m&=65535,m+=i*p,f+=m>>>16,m&=65535,f+=n*d+r*c+s*p+i*o,l(b<<16|k,(f&=65535)<<16|m,this.unsigned)},S.mul=S.multiply,S.divide=function(e){if(a(e)||(e=h(e)),e.isZero())throw Error("division by zero");if(t)return this.unsigned||-2147483648!==this.high||-1!==e.low||-1!==e.high?l((this.unsigned?t.div_u:t.div_s)(this.low,this.high,e.low,e.high),t.get_high(),this.unsigned):this;if(this.isZero())return this.unsigned?b:y;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return b;if(e.gt(this.shru(1)))return N;s=b}else{if(this.eq(T))return e.eq(k)||e.eq(v)?T:e.eq(T)?k:(n=this.shr(1).div(e).shl(1)).eq(y)?e.isNegative()?k:v:(r=this.sub(e.mul(n)),s=n.add(r.div(e)));if(e.eq(T))return this.unsigned?b:y;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();s=y}for(r=this;r.gte(e);){for(var n,r,s,i=Math.ceil(Math.log(n=Math.max(1,Math.floor(r.toNumber()/e.toNumber())))/Math.LN2),o=i<=48?1:p(2,i-48),c=u(n),d=c.mul(e);d.isNegative()||d.gt(r);)n-=o,d=(c=u(n,this.unsigned)).mul(e);c.isZero()&&(c=k),s=s.add(c),r=r.sub(d)}return s},S.div=S.divide,S.modulo=function(e){return(a(e)||(e=h(e)),t)?l((this.unsigned?t.rem_u:t.rem_s)(this.low,this.high,e.low,e.high),t.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},S.mod=S.modulo,S.rem=S.modulo,S.not=function(){return l(~this.low,~this.high,this.unsigned)},S.and=function(e){return a(e)||(e=h(e)),l(this.low&e.low,this.high&e.high,this.unsigned)},S.or=function(e){return a(e)||(e=h(e)),l(this.low|e.low,this.high|e.high,this.unsigned)},S.xor=function(e){return a(e)||(e=h(e)),l(this.low^e.low,this.high^e.high,this.unsigned)},S.shiftLeft=function(e){return(a(e)&&(e=e.toInt()),0==(e&=63))?this:e<32?l(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):l(0,this.low<<e-32,this.unsigned)},S.shl=S.shiftLeft,S.shiftRight=function(e){return(a(e)&&(e=e.toInt()),0==(e&=63))?this:e<32?l(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):l(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},S.shr=S.shiftRight,S.shiftRightUnsigned=function(e){if(a(e)&&(e=e.toInt()),0==(e&=63))return this;var t=this.high;return e<32?l(this.low>>>e|t<<32-e,t>>>e,this.unsigned):32===e?l(t,0,this.unsigned):l(t>>>e-32,0,this.unsigned)},S.shru=S.shiftRightUnsigned,S.shr_u=S.shiftRightUnsigned,S.toSigned=function(){return this.unsigned?l(this.low,this.high,!1):this},S.toUnsigned=function(){return this.unsigned?this:l(this.low,this.high,!0)},S.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},S.toBytesLE=function(){var e=this.high,t=this.low;return[255&t,t>>>8&255,t>>>16&255,t>>>24,255&e,e>>>8&255,e>>>16&255,e>>>24]},S.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,255&e,t>>>24,t>>>16&255,t>>>8&255,255&t]},r.fromBytes=function(e,t,n){return n?r.fromBytesLE(e,t):r.fromBytesBE(e,t)},r.fromBytesLE=function(e,t){return new r(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},r.fromBytesBE=function(e,t){return new r(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)}},3454:function(e,t,n){"use strict";var r,a;e.exports=(null==(r=n.g.process)?void 0:r.env)&&"object"==typeof(null==(a=n.g.process)?void 0:a.env)?n.g.process:n(7663)},1876:function(e){!function(){var t={675:function(e,t){"use strict";t.byteLength=function(e){var t=u(e),n=t[0],r=t[1];return(n+r)*3/4-r},t.toByteArray=function(e){var t,n,s,i,o=u(e),l=o[0],p=o[1],c=new a((l+p)*3/4-p),h=0,d=p>0?l-4:l;for(n=0;n<d;n+=4)t=r[e.charCodeAt(n)]<<18|r[e.charCodeAt(n+1)]<<12|r[e.charCodeAt(n+2)]<<6|r[e.charCodeAt(n+3)],c[h++]=t>>16&255,c[h++]=t>>8&255,c[h++]=255&t;return 2===p&&(t=r[e.charCodeAt(n)]<<2|r[e.charCodeAt(n+1)]>>4,c[h++]=255&t),1===p&&(t=r[e.charCodeAt(n)]<<10|r[e.charCodeAt(n+1)]<<4|r[e.charCodeAt(n+2)]>>2,c[h++]=t>>8&255,c[h++]=255&t),c},t.fromByteArray=function(e){for(var t,r=e.length,a=r%3,s=[],i=0,o=r-a;i<o;i+=16383)s.push(p(e,i,i+16383>o?o:i+16383));return 1===a?s.push(n[(t=e[r-1])>>2]+n[t<<4&63]+"=="):2===a&&s.push(n[(t=(e[r-2]<<8)+e[r-1])>>10]+n[t>>4&63]+n[t<<2&63]+"="),s.join("")};for(var n=[],r=[],a="undefined"!=typeof Uint8Array?Uint8Array:Array,s="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",i=0,o=s.length;i<o;++i)n[i]=s[i],r[s.charCodeAt(i)]=i;function u(e){var t=e.length;if(t%4>0)throw Error("Invalid string. Length must be a multiple of 4");var n=e.indexOf("=");-1===n&&(n=t);var r=n===t?0:4-n%4;return[n,r]}function l(e){return n[e>>18&63]+n[e>>12&63]+n[e>>6&63]+n[63&e]}function p(e,t,n){for(var r,a=[],s=t;s<n;s+=3)a.push(l(r=(e[s]<<16&16711680)+(e[s+1]<<8&65280)+(255&e[s+2])));return a.join("")}r["-".charCodeAt(0)]=62,r["_".charCodeAt(0)]=63},72:function(e,t,n){"use strict";/*!
 * The buffer module from node.js, for the browser.
 *
 * @author   Feross Aboukhadijeh <https://feross.org>
 * @license  MIT
 */ var r=n(675),a=n(783),s="function"==typeof Symbol&&"function"==typeof Symbol.for?Symbol.for("nodejs.util.inspect.custom"):null;function i(e){if(e>2147483647)throw RangeError('The value "'+e+'" is invalid for option "size"');var t=new Uint8Array(e);return Object.setPrototypeOf(t,o.prototype),t}function o(e,t,n){if("number"==typeof e){if("string"==typeof t)throw TypeError('The "string" argument must be of type string. Received type number');return p(e)}return u(e,t,n)}function u(e,t,n){if("string"==typeof e)return function(e,t){if(("string"!=typeof t||""===t)&&(t="utf8"),!o.isEncoding(t))throw TypeError("Unknown encoding: "+t);var n=0|d(e,t),r=i(n),a=r.write(e,t);return a!==n&&(r=r.slice(0,a)),r}(e,t);if(ArrayBuffer.isView(e))return c(e);if(null==e)throw TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type "+typeof e);if(V(e,ArrayBuffer)||e&&V(e.buffer,ArrayBuffer)||"undefined"!=typeof SharedArrayBuffer&&(V(e,SharedArrayBuffer)||e&&V(e.buffer,SharedArrayBuffer)))return function e(t,n,r){var a;if(n<0||t.byteLength<n)throw RangeError('"offset" is outside of buffer bounds');if(t.byteLength<n+(r||0))throw RangeError('"length" is outside of buffer bounds');return Object.setPrototypeOf(a=void 0===n&&void 0===r?new Uint8Array(t):void 0===r?new Uint8Array(t,n):new Uint8Array(t,n,r),o.prototype),a}(e,t,n);if("number"==typeof e)throw TypeError('The "value" argument must not be of type number. Received type number');var r=e.valueOf&&e.valueOf();if(null!=r&&r!==e)return o.from(r,t,n);var a=function(e){if(o.isBuffer(e)){var t,n=0|h(e.length),r=i(n);return 0===r.length||e.copy(r,0,0,n),r}if(void 0!==e.length){return"number"!=typeof e.length||(t=e.length,t!=t)?i(0):c(e)}if("Buffer"===e.type&&Array.isArray(e.data))return c(e.data)}(e);if(a)return a;if("undefined"!=typeof Symbol&&null!=Symbol.toPrimitive&&"function"==typeof e[Symbol.toPrimitive])return o.from(e[Symbol.toPrimitive]("string"),t,n);throw TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type "+typeof e)}function l(e){if("number"!=typeof e)throw TypeError('"size" argument must be of type number');if(e<0)throw RangeError('The value "'+e+'" is invalid for option "size"')}function p(e){return l(e),i(e<0?0:0|h(e))}function c(e){for(var t=e.length<0?0:0|h(e.length),n=i(t),r=0;r<t;r+=1)n[r]=255&e[r];return n}t.Buffer=o,t.SlowBuffer=function(e){return+e!=e&&(e=0),o.alloc(+e)},t.INSPECT_MAX_BYTES=50,t.kMaxLength=2147483647,o.TYPED_ARRAY_SUPPORT=function(){try{var e=new Uint8Array(1),t={foo:function(){return 42}};return Object.setPrototypeOf(t,Uint8Array.prototype),Object.setPrototypeOf(e,t),42===e.foo()}catch(n){return!1}}(),o.TYPED_ARRAY_SUPPORT||"undefined"==typeof console||"function"!=typeof console.error||console.error("This browser lacks typed array (Uint8Array) support which is required by `buffer` v5.x. Use `buffer` v4.x if you require old browser support."),Object.defineProperty(o.prototype,"parent",{enumerable:!0,get:function(){if(o.isBuffer(this))return this.buffer}}),Object.defineProperty(o.prototype,"offset",{enumerable:!0,get:function(){if(o.isBuffer(this))return this.byteOffset}}),o.poolSize=8192,o.from=function(e,t,n){return u(e,t,n)},Object.setPrototypeOf(o.prototype,Uint8Array.prototype),Object.setPrototypeOf(o,Uint8Array),o.alloc=function(e,t,n){var r,a,s;return(l(e),e<=0)?i(e):void 0!==t?"string"==typeof n?i(e).fill(t,n):i(e).fill(t):i(e)},o.allocUnsafe=function(e){return p(e)},o.allocUnsafeSlow=function(e){return p(e)};function h(e){if(e>=2147483647)throw RangeError("Attempt to allocate Buffer larger than maximum size: 0x"+2147483647..toString(16)+" bytes");return 0|e}function d(e,t){if(o.isBuffer(e))return e.length;if(ArrayBuffer.isView(e)||V(e,ArrayBuffer))return e.byteLength;if("string"!=typeof e)throw TypeError('The "string" argument must be one of type string, Buffer, or ArrayBuffer. Received type '+typeof e);var n=e.length,r=arguments.length>2&&!0===arguments[2];if(!r&&0===n)return 0;for(var a=!1;;)switch(t){case"ascii":case"latin1":case"binary":return n;case"utf8":case"utf-8":return O(e).length;case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return 2*n;case"hex":return n>>>1;case"base64":return R(e).length;default:if(a)return r?-1:O(e).length;t=(""+t).toLowerCase(),a=!0}}function f(e,t,n){var r=!1;if((void 0===t||t<0)&&(t=0),t>this.length||((void 0===n||n>this.length)&&(n=this.length),n<=0||(n>>>=0)<=(t>>>=0)))return"";for(e||(e="utf8");;)switch(e){case"hex":return _(this,t,n);case"utf8":case"utf-8":return T(this,t,n);case"ascii":return S(this,t,n);case"latin1":case"binary":return I(this,t,n);case"base64":return w(this,t,n);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return E(this,t,n);default:if(r)throw TypeError("Unknown encoding: "+e);e=(e+"").toLowerCase(),r=!0}}function m(e,t,n){var r=e[t];e[t]=e[n],e[n]=r}function g(e,t,n,r,a){var s;if(0===e.length)return -1;if("string"==typeof n?(r=n,n=0):n>2147483647?n=2147483647:n<-2147483648&&(n=-2147483648),s=n=+n,s!=s&&(n=a?0:e.length-1),n<0&&(n=e.length+n),n>=e.length){if(a)return -1;n=e.length-1}else if(n<0){if(!a)return -1;n=0}if("string"==typeof t&&(t=o.from(t,r)),o.isBuffer(t))return 0===t.length?-1:y(e,t,n,r,a);if("number"==typeof t)return(t&=255,"function"==typeof Uint8Array.prototype.indexOf)?a?Uint8Array.prototype.indexOf.call(e,t,n):Uint8Array.prototype.lastIndexOf.call(e,t,n):y(e,[t],n,r,a);throw TypeError("val must be string, number or Buffer")}function y(e,t,n,r,a){var s,i=1,o=e.length,u=t.length;if(void 0!==r&&("ucs2"===(r=String(r).toLowerCase())||"ucs-2"===r||"utf16le"===r||"utf-16le"===r)){if(e.length<2||t.length<2)return -1;i=2,o/=2,u/=2,n/=2}function l(e,t){return 1===i?e[t]:e.readUInt16BE(t*i)}if(a){var p=-1;for(s=n;s<o;s++)if(l(e,s)===l(t,-1===p?0:s-p)){if(-1===p&&(p=s),s-p+1===u)return p*i}else -1!==p&&(s-=s-p),p=-1}else for(n+u>o&&(n=o-u),s=n;s>=0;s--){for(var c=!0,h=0;h<u;h++)if(l(e,s+h)!==l(t,h)){c=!1;break}if(c)return s}return -1}function b(e,t,n,r){n=Number(n)||0;var a=e.length-n;r?(r=Number(r))>a&&(r=a):r=a;var s=t.length;r>s/2&&(r=s/2);for(var i=0;i<r;++i){var o,u=parseInt(t.substr(2*i,2),16);if(o=u,o!=o)break;e[n+i]=u}return i}function k(e,t,n,r){return C(O(t,e.length-n),e,n,r)}function N(e,t,n,r){return C(function(e){for(var t=[],n=0;n<e.length;++n)t.push(255&e.charCodeAt(n));return t}(t),e,n,r)}function v(e,t,n,r){return C(R(t),e,n,r)}function x(e,t,n,r){return C(function(e,t){for(var n,r,a,s=[],i=0;i<e.length&&!((t-=2)<0);++i)r=(n=e.charCodeAt(i))>>8,a=n%256,s.push(a),s.push(r);return s}(t,e.length-n),e,n,r)}function w(e,t,n){return 0===t&&n===e.length?r.fromByteArray(e):r.fromByteArray(e.slice(t,n))}function T(e,t,n){n=Math.min(e.length,n);for(var r=[],a=t;a<n;){var s,i,o,u,l=e[a],p=null,c=l>239?4:l>223?3:l>191?2:1;if(a+c<=n)switch(c){case 1:l<128&&(p=l);break;case 2:(192&(s=e[a+1]))==128&&(u=(31&l)<<6|63&s)>127&&(p=u);break;case 3:s=e[a+1],i=e[a+2],(192&s)==128&&(192&i)==128&&(u=(15&l)<<12|(63&s)<<6|63&i)>2047&&(u<55296||u>57343)&&(p=u);break;case 4:s=e[a+1],i=e[a+2],o=e[a+3],(192&s)==128&&(192&i)==128&&(192&o)==128&&(u=(15&l)<<18|(63&s)<<12|(63&i)<<6|63&o)>65535&&u<1114112&&(p=u)}null===p?(p=65533,c=1):p>65535&&(p-=65536,r.push(p>>>10&1023|55296),p=56320|1023&p),r.push(p),a+=c}return function(e){var t=e.length;if(t<=4096)return String.fromCharCode.apply(String,e);for(var n="",r=0;r<t;)n+=String.fromCharCode.apply(String,e.slice(r,r+=4096));return n}(r)}function S(e,t,n){var r="";n=Math.min(e.length,n);for(var a=t;a<n;++a)r+=String.fromCharCode(127&e[a]);return r}function I(e,t,n){var r="";n=Math.min(e.length,n);for(var a=t;a<n;++a)r+=String.fromCharCode(e[a]);return r}function _(e,t,n){var r=e.length;(!t||t<0)&&(t=0),(!n||n<0||n>r)&&(n=r);for(var a="",s=t;s<n;++s)a+=P[e[s]];return a}function E(e,t,n){for(var r=e.slice(t,n),a="",s=0;s<r.length;s+=2)a+=String.fromCharCode(r[s]+256*r[s+1]);return a}function A(e,t,n){if(e%1!=0||e<0)throw RangeError("offset is not uint");if(e+t>n)throw RangeError("Trying to access beyond buffer length")}function M(e,t,n,r,a,s){if(!o.isBuffer(e))throw TypeError('"buffer" argument must be a Buffer instance');if(t>a||t<s)throw RangeError('"value" argument is out of bounds');if(n+r>e.length)throw RangeError("Index out of range")}function D(e,t,n,r,a,s){if(n+r>e.length||n<0)throw RangeError("Index out of range")}function $(e,t,n,r,s){return t=+t,n>>>=0,s||D(e,t,n,4,34028234663852886e22,-34028234663852886e22),a.write(e,t,n,r,23,4),n+4}function F(e,t,n,r,s){return t=+t,n>>>=0,s||D(e,t,n,8,17976931348623157e292,-17976931348623157e292),a.write(e,t,n,r,52,8),n+8}o.isBuffer=function(e){return null!=e&&!0===e._isBuffer&&e!==o.prototype},o.compare=function(e,t){if(V(e,Uint8Array)&&(e=o.from(e,e.offset,e.byteLength)),V(t,Uint8Array)&&(t=o.from(t,t.offset,t.byteLength)),!o.isBuffer(e)||!o.isBuffer(t))throw TypeError('The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array');if(e===t)return 0;for(var n=e.length,r=t.length,a=0,s=Math.min(n,r);a<s;++a)if(e[a]!==t[a]){n=e[a],r=t[a];break}return n<r?-1:r<n?1:0},o.isEncoding=function(e){switch(String(e).toLowerCase()){case"hex":case"utf8":case"utf-8":case"ascii":case"latin1":case"binary":case"base64":case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return!0;default:return!1}},o.concat=function(e,t){if(!Array.isArray(e))throw TypeError('"list" argument must be an Array of Buffers');if(0===e.length)return o.alloc(0);if(void 0===t)for(n=0,t=0;n<e.length;++n)t+=e[n].length;var n,r=o.allocUnsafe(t),a=0;for(n=0;n<e.length;++n){var s=e[n];if(V(s,Uint8Array)&&(s=o.from(s)),!o.isBuffer(s))throw TypeError('"list" argument must be an Array of Buffers');s.copy(r,a),a+=s.length}return r},o.byteLength=d,o.prototype._isBuffer=!0,o.prototype.swap16=function(){var e=this.length;if(e%2!=0)throw RangeError("Buffer size must be a multiple of 16-bits");for(var t=0;t<e;t+=2)m(this,t,t+1);return this},o.prototype.swap32=function(){var e=this.length;if(e%4!=0)throw RangeError("Buffer size must be a multiple of 32-bits");for(var t=0;t<e;t+=4)m(this,t,t+3),m(this,t+1,t+2);return this},o.prototype.swap64=function(){var e=this.length;if(e%8!=0)throw RangeError("Buffer size must be a multiple of 64-bits");for(var t=0;t<e;t+=8)m(this,t,t+7),m(this,t+1,t+6),m(this,t+2,t+5),m(this,t+3,t+4);return this},o.prototype.toString=function(){var e=this.length;return 0===e?"":0===arguments.length?T(this,0,e):f.apply(this,arguments)},o.prototype.toLocaleString=o.prototype.toString,o.prototype.equals=function(e){if(!o.isBuffer(e))throw TypeError("Argument must be a Buffer");return this===e||0===o.compare(this,e)},o.prototype.inspect=function(){var e="",n=t.INSPECT_MAX_BYTES;return e=this.toString("hex",0,n).replace(/(.{2})/g,"$1 ").trim(),this.length>n&&(e+=" ... "),"<Buffer "+e+">"},s&&(o.prototype[s]=o.prototype.inspect),o.prototype.compare=function(e,t,n,r,a){if(V(e,Uint8Array)&&(e=o.from(e,e.offset,e.byteLength)),!o.isBuffer(e))throw TypeError('The "target" argument must be one of type Buffer or Uint8Array. Received type '+typeof e);if(void 0===t&&(t=0),void 0===n&&(n=e?e.length:0),void 0===r&&(r=0),void 0===a&&(a=this.length),t<0||n>e.length||r<0||a>this.length)throw RangeError("out of range index");if(r>=a&&t>=n)return 0;if(r>=a)return -1;if(t>=n)return 1;if(t>>>=0,n>>>=0,r>>>=0,a>>>=0,this===e)return 0;for(var s=a-r,i=n-t,u=Math.min(s,i),l=this.slice(r,a),p=e.slice(t,n),c=0;c<u;++c)if(l[c]!==p[c]){s=l[c],i=p[c];break}return s<i?-1:i<s?1:0},o.prototype.includes=function(e,t,n){return -1!==this.indexOf(e,t,n)},o.prototype.indexOf=function(e,t,n){return g(this,e,t,n,!0)},o.prototype.lastIndexOf=function(e,t,n){return g(this,e,t,n,!1)},o.prototype.write=function(e,t,n,r){if(void 0===t)r="utf8",n=this.length,t=0;else if(void 0===n&&"string"==typeof t)r=t,n=this.length,t=0;else if(isFinite(t))t>>>=0,isFinite(n)?(n>>>=0,void 0===r&&(r="utf8")):(r=n,n=void 0);else throw Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");var a,s,i,o,u=this.length-t;if((void 0===n||n>u)&&(n=u),e.length>0&&(n<0||t<0)||t>this.length)throw RangeError("Attempt to write outside buffer bounds");r||(r="utf8");for(var l=!1;;)switch(r){case"hex":return b(this,e,t,n);case"utf8":case"utf-8":return k(this,e,t,n);case"ascii":return N(this,e,t,n);case"latin1":case"binary":return i=t,N(this,e,i,o=n);case"base64":return v(this,e,t,n);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return x(this,e,t,n);default:if(l)throw TypeError("Unknown encoding: "+r);r=(""+r).toLowerCase(),l=!0}},o.prototype.toJSON=function(){return{type:"Buffer",data:Array.prototype.slice.call(this._arr||this,0)}},o.prototype.slice=function(e,t){var n=this.length;e=~~e,t=void 0===t?n:~~t,e<0?(e+=n)<0&&(e=0):e>n&&(e=n),t<0?(t+=n)<0&&(t=0):t>n&&(t=n),t<e&&(t=e);var r=this.subarray(e,t);return Object.setPrototypeOf(r,o.prototype),r},o.prototype.readUIntLE=function(e,t,n){e>>>=0,t>>>=0,n||A(e,t,this.length);for(var r=this[e],a=1,s=0;++s<t&&(a*=256);)r+=this[e+s]*a;return r},o.prototype.readUIntBE=function(e,t,n){e>>>=0,t>>>=0,n||A(e,t,this.length);for(var r=this[e+--t],a=1;t>0&&(a*=256);)r+=this[e+--t]*a;return r},o.prototype.readUInt8=function(e,t){return e>>>=0,t||A(e,1,this.length),this[e]},o.prototype.readUInt16LE=function(e,t){return e>>>=0,t||A(e,2,this.length),this[e]|this[e+1]<<8},o.prototype.readUInt16BE=function(e,t){return e>>>=0,t||A(e,2,this.length),this[e]<<8|this[e+1]},o.prototype.readUInt32LE=function(e,t){return e>>>=0,t||A(e,4,this.length),(this[e]|this[e+1]<<8|this[e+2]<<16)+16777216*this[e+3]},o.prototype.readUInt32BE=function(e,t){return e>>>=0,t||A(e,4,this.length),16777216*this[e]+(this[e+1]<<16|this[e+2]<<8|this[e+3])},o.prototype.readIntLE=function(e,t,n){e>>>=0,t>>>=0,n||A(e,t,this.length);for(var r=this[e],a=1,s=0;++s<t&&(a*=256);)r+=this[e+s]*a;return r>=(a*=128)&&(r-=Math.pow(2,8*t)),r},o.prototype.readIntBE=function(e,t,n){e>>>=0,t>>>=0,n||A(e,t,this.length);for(var r=t,a=1,s=this[e+--r];r>0&&(a*=256);)s+=this[e+--r]*a;return s>=(a*=128)&&(s-=Math.pow(2,8*t)),s},o.prototype.readInt8=function(e,t){return(e>>>=0,t||A(e,1,this.length),128&this[e])?-((255-this[e]+1)*1):this[e]},o.prototype.readInt16LE=function(e,t){e>>>=0,t||A(e,2,this.length);var n=this[e]|this[e+1]<<8;return 32768&n?4294901760|n:n},o.prototype.readInt16BE=function(e,t){e>>>=0,t||A(e,2,this.length);var n=this[e+1]|this[e]<<8;return 32768&n?4294901760|n:n},o.prototype.readInt32LE=function(e,t){return e>>>=0,t||A(e,4,this.length),this[e]|this[e+1]<<8|this[e+2]<<16|this[e+3]<<24},o.prototype.readInt32BE=function(e,t){return e>>>=0,t||A(e,4,this.length),this[e]<<24|this[e+1]<<16|this[e+2]<<8|this[e+3]},o.prototype.readFloatLE=function(e,t){return e>>>=0,t||A(e,4,this.length),a.read(this,e,!0,23,4)},o.prototype.readFloatBE=function(e,t){return e>>>=0,t||A(e,4,this.length),a.read(this,e,!1,23,4)},o.prototype.readDoubleLE=function(e,t){return e>>>=0,t||A(e,8,this.length),a.read(this,e,!0,52,8)},o.prototype.readDoubleBE=function(e,t){return e>>>=0,t||A(e,8,this.length),a.read(this,e,!1,52,8)},o.prototype.writeUIntLE=function(e,t,n,r){if(e=+e,t>>>=0,n>>>=0,!r){var a=Math.pow(2,8*n)-1;M(this,e,t,n,a,0)}var s=1,i=0;for(this[t]=255&e;++i<n&&(s*=256);)this[t+i]=e/s&255;return t+n},o.prototype.writeUIntBE=function(e,t,n,r){if(e=+e,t>>>=0,n>>>=0,!r){var a=Math.pow(2,8*n)-1;M(this,e,t,n,a,0)}var s=n-1,i=1;for(this[t+s]=255&e;--s>=0&&(i*=256);)this[t+s]=e/i&255;return t+n},o.prototype.writeUInt8=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,1,255,0),this[t]=255&e,t+1},o.prototype.writeUInt16LE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,2,65535,0),this[t]=255&e,this[t+1]=e>>>8,t+2},o.prototype.writeUInt16BE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,2,65535,0),this[t]=e>>>8,this[t+1]=255&e,t+2},o.prototype.writeUInt32LE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,4,4294967295,0),this[t+3]=e>>>24,this[t+2]=e>>>16,this[t+1]=e>>>8,this[t]=255&e,t+4},o.prototype.writeUInt32BE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,4,4294967295,0),this[t]=e>>>24,this[t+1]=e>>>16,this[t+2]=e>>>8,this[t+3]=255&e,t+4},o.prototype.writeIntLE=function(e,t,n,r){if(e=+e,t>>>=0,!r){var a=Math.pow(2,8*n-1);M(this,e,t,n,a-1,-a)}var s=0,i=1,o=0;for(this[t]=255&e;++s<n&&(i*=256);)e<0&&0===o&&0!==this[t+s-1]&&(o=1),this[t+s]=(e/i>>0)-o&255;return t+n},o.prototype.writeIntBE=function(e,t,n,r){if(e=+e,t>>>=0,!r){var a=Math.pow(2,8*n-1);M(this,e,t,n,a-1,-a)}var s=n-1,i=1,o=0;for(this[t+s]=255&e;--s>=0&&(i*=256);)e<0&&0===o&&0!==this[t+s+1]&&(o=1),this[t+s]=(e/i>>0)-o&255;return t+n},o.prototype.writeInt8=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,1,127,-128),e<0&&(e=255+e+1),this[t]=255&e,t+1},o.prototype.writeInt16LE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,2,32767,-32768),this[t]=255&e,this[t+1]=e>>>8,t+2},o.prototype.writeInt16BE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,2,32767,-32768),this[t]=e>>>8,this[t+1]=255&e,t+2},o.prototype.writeInt32LE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,4,2147483647,-2147483648),this[t]=255&e,this[t+1]=e>>>8,this[t+2]=e>>>16,this[t+3]=e>>>24,t+4},o.prototype.writeInt32BE=function(e,t,n){return e=+e,t>>>=0,n||M(this,e,t,4,2147483647,-2147483648),e<0&&(e=4294967295+e+1),this[t]=e>>>24,this[t+1]=e>>>16,this[t+2]=e>>>8,this[t+3]=255&e,t+4},o.prototype.writeFloatLE=function(e,t,n){return $(this,e,t,!0,n)},o.prototype.writeFloatBE=function(e,t,n){return $(this,e,t,!1,n)},o.prototype.writeDoubleLE=function(e,t,n){return F(this,e,t,!0,n)},o.prototype.writeDoubleBE=function(e,t,n){return F(this,e,t,!1,n)},o.prototype.copy=function(e,t,n,r){if(!o.isBuffer(e))throw TypeError("argument should be a Buffer");if(n||(n=0),r||0===r||(r=this.length),t>=e.length&&(t=e.length),t||(t=0),r>0&&r<n&&(r=n),r===n||0===e.length||0===this.length)return 0;if(t<0)throw RangeError("targetStart out of bounds");if(n<0||n>=this.length)throw RangeError("Index out of range");if(r<0)throw RangeError("sourceEnd out of bounds");r>this.length&&(r=this.length),e.length-t<r-n&&(r=e.length-t+n);var a=r-n;if(this===e&&"function"==typeof Uint8Array.prototype.copyWithin)this.copyWithin(t,n,r);else if(this===e&&n<t&&t<r)for(var s=a-1;s>=0;--s)e[s+t]=this[s+n];else Uint8Array.prototype.set.call(e,this.subarray(n,r),t);return a},o.prototype.fill=function(e,t,n,r){if("string"==typeof e){if("string"==typeof t?(r=t,t=0,n=this.length):"string"==typeof n&&(r=n,n=this.length),void 0!==r&&"string"!=typeof r)throw TypeError("encoding must be a string");if("string"==typeof r&&!o.isEncoding(r))throw TypeError("Unknown encoding: "+r);if(1===e.length){var a,s=e.charCodeAt(0);("utf8"===r&&s<128||"latin1"===r)&&(e=s)}}else"number"==typeof e?e&=255:"boolean"==typeof e&&(e=Number(e));if(t<0||this.length<t||this.length<n)throw RangeError("Out of range index");if(n<=t)return this;if(t>>>=0,n=void 0===n?this.length:n>>>0,e||(e=0),"number"==typeof e)for(a=t;a<n;++a)this[a]=e;else{var i=o.isBuffer(e)?e:o.from(e,r),u=i.length;if(0===u)throw TypeError('The value "'+e+'" is invalid for argument "value"');for(a=0;a<n-t;++a)this[a+t]=i[a%u]}return this};var B=/[^+/0-9A-Za-z-_]/g;function O(e,t){t=t||1/0;for(var n,r=e.length,a=null,s=[],i=0;i<r;++i){if((n=e.charCodeAt(i))>55295&&n<57344){if(!a){if(n>56319||i+1===r){(t-=3)>-1&&s.push(239,191,189);continue}a=n;continue}if(n<56320){(t-=3)>-1&&s.push(239,191,189),a=n;continue}n=(a-55296<<10|n-56320)+65536}else a&&(t-=3)>-1&&s.push(239,191,189);if(a=null,n<128){if((t-=1)<0)break;s.push(n)}else if(n<2048){if((t-=2)<0)break;s.push(n>>6|192,63&n|128)}else if(n<65536){if((t-=3)<0)break;s.push(n>>12|224,n>>6&63|128,63&n|128)}else if(n<1114112){if((t-=4)<0)break;s.push(n>>18|240,n>>12&63|128,n>>6&63|128,63&n|128)}else throw Error("Invalid code point")}return s}function R(e){return r.toByteArray(function(e){if((e=(e=e.split("=")[0]).trim().replace(B,"")).length<2)return"";for(;e.length%4!=0;)e+="=";return e}(e))}function C(e,t,n,r){for(var a=0;a<r&&!(a+n>=t.length)&&!(a>=e.length);++a)t[a+n]=e[a];return a}function V(e,t){return e instanceof t||null!=e&&null!=e.constructor&&null!=e.constructor.name&&e.constructor.name===t.name}var P=function(){for(var e="0123456789abcdef",t=Array(256),n=0;n<16;++n)for(var r=16*n,a=0;a<16;++a)t[r+a]=e[n]+e[a];return t}()},783:function(e,t){/*! ieee754. BSD-3-Clause License. Feross Aboukhadijeh <https://feross.org/opensource> */ t.read=function(e,t,n,r,a){var s,i,o=8*a-r-1,u=(1<<o)-1,l=u>>1,p=-7,c=n?a-1:0,h=n?-1:1,d=e[t+c];for(c+=h,s=d&(1<<-p)-1,d>>=-p,p+=o;p>0;s=256*s+e[t+c],c+=h,p-=8);for(i=s&(1<<-p)-1,s>>=-p,p+=r;p>0;i=256*i+e[t+c],c+=h,p-=8);if(0===s)s=1-l;else{if(s===u)return i?NaN:(d?-1:1)*(1/0);i+=Math.pow(2,r),s-=l}return(d?-1:1)*i*Math.pow(2,s-r)},t.write=function(e,t,n,r,a,s){var i,o,u,l=8*s-a-1,p=(1<<l)-1,c=p>>1,h=23===a?5960464477539062e-23:0,d=r?0:s-1,f=r?1:-1,m=t<0||0===t&&1/t<0?1:0;for(isNaN(t=Math.abs(t))||t===1/0?(o=isNaN(t)?1:0,i=p):(i=Math.floor(Math.log(t)/Math.LN2),t*(u=Math.pow(2,-i))<1&&(i--,u*=2),i+c>=1?t+=h/u:t+=h*Math.pow(2,1-c),t*u>=2&&(i++,u/=2),i+c>=p?(o=0,i=p):i+c>=1?(o=(t*u-1)*Math.pow(2,a),i+=c):(o=t*Math.pow(2,c-1)*Math.pow(2,a),i=0));a>=8;e[n+d]=255&o,d+=f,o/=256,a-=8);for(i=i<<a|o,l+=a;l>0;e[n+d]=255&i,d+=f,i/=256,l-=8);e[n+d-f]|=128*m}}},n={};function r(e){var a=n[e];if(void 0!==a)return a.exports;var s=n[e]={exports:{}},i=!0;try{t[e](s,s.exports,r),i=!1}finally{i&&delete n[e]}return s.exports}r.ab="//";var a=r(72);e.exports=a}()},7663:function(e){!function(){var t={229:function(e){var t,n,r,a=e.exports={};function s(){throw Error("setTimeout has not been defined")}function i(){throw Error("clearTimeout has not been defined")}function o(e){if(t===setTimeout)return setTimeout(e,0);if((t===s||!t)&&setTimeout)return t=setTimeout,setTimeout(e,0);try{return t(e,0)}catch(r){try{return t.call(null,e,0)}catch(n){return t.call(this,e,0)}}}!function(){try{t="function"==typeof setTimeout?setTimeout:s}catch(e){t=s}try{n="function"==typeof clearTimeout?clearTimeout:i}catch(r){n=i}}();var u=[],l=!1,p=-1;function c(){l&&r&&(l=!1,r.length?u=r.concat(u):p=-1,u.length&&h())}function h(){if(!l){var e=o(c);l=!0;for(var t=u.length;t;){for(r=u,u=[];++p<t;)r&&r[p].run();p=-1,t=u.length}r=null,l=!1,function(e){if(n===clearTimeout)return clearTimeout(e);if((n===i||!n)&&clearTimeout)return n=clearTimeout,clearTimeout(e);try{n(e)}catch(r){try{return n.call(null,e)}catch(t){return n.call(this,e)}}}(e)}}function d(e,t){this.fun=e,this.array=t}function f(){}a.nextTick=function(e){var t=Array(arguments.length-1);if(arguments.length>1)for(var n=1;n<arguments.length;n++)t[n-1]=arguments[n];u.push(new d(e,t)),1!==u.length||l||o(h)},d.prototype.run=function(){this.fun.apply(null,this.array)},a.title="browser",a.browser=!0,a.env={},a.argv=[],a.version="",a.versions={},a.on=f,a.addListener=f,a.once=f,a.off=f,a.removeListener=f,a.removeAllListeners=f,a.emit=f,a.prependListener=f,a.prependOnceListener=f,a.listeners=function(e){return[]},a.binding=function(e){throw Error("process.binding is not supported")},a.cwd=function(){return"/"},a.chdir=function(e){throw Error("process.chdir is not supported")},a.umask=function(){return 0}}},n={};function r(e){var a=n[e];if(void 0!==a)return a.exports;var s=n[e]={exports:{}},i=!0;try{t[e](s,s.exports,r),i=!1}finally{i&&delete n[e]}return s.exports}r.ab="//";var a=r(229);e.exports=a}()},6377:function(e,t,n){var r=n(4832),a=n(8652),s=n(801),i=n(2030),o=n(3618),u=n(9049),l=n(1971);l.alea=r,l.xor128=a,l.xorwow=s,l.xorshift7=i,l.xor4096=o,l.tychei=u,e.exports=l},4832:function(e,t,n){var r;!function(e,a,s){function i(e){var t,n=this,r=(t=4022871197,function(e){e=String(e);for(var n=0;n<e.length;n++){var r=.02519603282416938*(t+=e.charCodeAt(n));t=r>>>0,r-=t,r*=t,t=r>>>0,r-=t,t+=4294967296*r}return(t>>>0)*23283064365386963e-26});n.next=function(){var e=2091639*n.s0+23283064365386963e-26*n.c;return n.s0=n.s1,n.s1=n.s2,n.s2=e-(n.c=0|e)},n.c=1,n.s0=r(" "),n.s1=r(" "),n.s2=r(" "),n.s0-=r(e),n.s0<0&&(n.s0+=1),n.s1-=r(e),n.s1<0&&(n.s1+=1),n.s2-=r(e),n.s2<0&&(n.s2+=1),r=null}function o(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function u(e,t){var n=new i(e),r=t&&t.state,a=n.next;return a.int32=function(){return 4294967296*n.next()|0},a.double=function(){return a()+(2097152*a()|0)*11102230246251565e-32},a.quick=a,r&&("object"==typeof r&&o(r,n),a.state=function(){return o(n,{})}),a}a&&a.exports?a.exports=u:n.amdD&&n.amdO?void 0!==(r=(function(){return u}).call(t,n,t,a))&&(a.exports=r):this.alea=u}(this,e=n.nmd(e),n.amdD)},9049:function(e,t,n){var r;!function(e,a,s){function i(e){var t=this,n="";t.next=function(){var e=t.b,n=t.c,r=t.d,a=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^a,a=a-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^a,t.a=a-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=0|e):n+=e;for(var r=0;r<n.length+20;r++)t.b^=0|n.charCodeAt(r),t.next()}function o(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function u(e,t){var n=new i(e),r=t&&t.state,a=function(){return(n.next()>>>0)/4294967296};return a.double=function(){do var e=n.next()>>>11,t=(e+(n.next()>>>0)/4294967296)/2097152;while(0===t);return t},a.int32=n.next,a.quick=a,r&&("object"==typeof r&&o(r,n),a.state=function(){return o(n,{})}),a}a&&a.exports?a.exports=u:n.amdD&&n.amdO?void 0!==(r=(function(){return u}).call(t,n,t,a))&&(a.exports=r):this.tychei=u}(this,e=n.nmd(e),n.amdD)},8652:function(e,t,n){var r;!function(e,a,s){function i(e){var t=this,n="";t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(0|e)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=0|n.charCodeAt(r),t.next()}function o(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function u(e,t){var n=new i(e),r=t&&t.state,a=function(){return(n.next()>>>0)/4294967296};return a.double=function(){do var e=n.next()>>>11,t=(e+(n.next()>>>0)/4294967296)/2097152;while(0===t);return t},a.int32=n.next,a.quick=a,r&&("object"==typeof r&&o(r,n),a.state=function(){return o(n,{})}),a}a&&a.exports?a.exports=u:n.amdD&&n.amdO?void 0!==(r=(function(){return u}).call(t,n,t,a))&&(a.exports=r):this.xor128=u}(this,e=n.nmd(e),n.amdD)},3618:function(e,t,n){var r;!function(e,a,s){function i(e){var t=this;t.next=function(){var e,n,r=t.w,a=t.X,s=t.i;return t.w=r=r+1640531527|0,n=a[s+34&127],e=a[s=s+1&127],n^=n<<13,e^=e<<17,n^=n>>>15,e^=e>>>12,n=a[s]=n^e,t.i=s,n+(r^r>>>16)|0},!function(e,t){var n,r,a,s,i,o=[],u=128;for(t===(0|t)?(r=t,t=null):(t+="\0",r=0,u=Math.max(u,t.length)),a=0,s=-32;s<u;++s)t&&(r^=t.charCodeAt((s+32)%t.length)),0===s&&(i=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,s>=0&&(i=i+1640531527|0,a=0==(n=o[127&s]^=r+i)?a+1:0);for(a>=128&&(o[127&(t&&t.length||0)]=-1),a=127,s=512;s>0;--s)r=o[a+34&127],n=o[a=a+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,o[a]=r^n;e.w=i,e.X=o,e.i=a}(t,e)}function o(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function u(e,t){null==e&&(e=+new Date);var n=new i(e),r=t&&t.state,a=function(){return(n.next()>>>0)/4294967296};return a.double=function(){do var e=n.next()>>>11,t=(e+(n.next()>>>0)/4294967296)/2097152;while(0===t);return t},a.int32=n.next,a.quick=a,r&&(r.X&&o(r,n),a.state=function(){return o(n,{})}),a}a&&a.exports?a.exports=u:n.amdD&&n.amdO?void 0!==(r=(function(){return u}).call(t,n,t,a))&&(a.exports=r):this.xor4096=u}(this,e=n.nmd(e),n.amdD)},2030:function(e,t,n){var r;!function(e,a,s){function i(e){var t=this;t.next=function(){var e,n,r=t.x,a=t.i;return e=r[a],e^=e>>>7,n=e^e<<24,n^=(e=r[a+1&7])^e>>>10,n^=(e=r[a+3&7])^e>>>3,n^=(e=r[a+4&7])^e<<7,e=r[a+7&7],e^=e<<13,n^=e^e<<9,r[a]=n,t.i=a+1&7,n},!function(e,t){var n,r=[];if(t===(0|t))r[0]=t;else for(n=0,t=""+t;n<t.length;++n)r[7&n]=r[7&n]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&0===r[n];++n);for(8==n?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}(t,e)}function o(e,t){return t.x=e.x.slice(),t.i=e.i,t}function u(e,t){null==e&&(e=+new Date);var n=new i(e),r=t&&t.state,a=function(){return(n.next()>>>0)/4294967296};return a.double=function(){do var e=n.next()>>>11,t=(e+(n.next()>>>0)/4294967296)/2097152;while(0===t);return t},a.int32=n.next,a.quick=a,r&&(r.x&&o(r,n),a.state=function(){return o(n,{})}),a}a&&a.exports?a.exports=u:n.amdD&&n.amdO?void 0!==(r=(function(){return u}).call(t,n,t,a))&&(a.exports=r):this.xorshift7=u}(this,e=n.nmd(e),n.amdD)},801:function(e,t,n){var r;!function(e,a,s){function i(e){var t=this,n="";t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^(e^e<<1))|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(0|e)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=0|n.charCodeAt(r),r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function o(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function u(e,t){var n=new i(e),r=t&&t.state,a=function(){return(n.next()>>>0)/4294967296};return a.double=function(){do var e=n.next()>>>11,t=(e+(n.next()>>>0)/4294967296)/2097152;while(0===t);return t},a.int32=n.next,a.quick=a,r&&("object"==typeof r&&o(r,n),a.state=function(){return o(n,{})}),a}a&&a.exports?a.exports=u:n.amdD&&n.amdO?void 0!==(r=(function(){return u}).call(t,n,t,a))&&(a.exports=r):this.xorwow=u}(this,e=n.nmd(e),n.amdD)},1971:function(e,t,n){var r;!function(a,s,i){var o,u=i.pow(256,6),l=i.pow(2,52),p=2*l;function c(e,t,n){var r=[],c=f(function e(t,n){var r,a=[],s=typeof t;if(n&&"object"==s)for(r in t)try{a.push(e(t[r],n-1))}catch(i){}return a.length?a:"string"==s?t:t+"\0"}((t=!0==t?{entropy:!0}:t||{}).entropy?[e,m(s)]:null==e?function(){try{var e;return o&&(e=o.randomBytes)?e=e(256):(e=new Uint8Array(256),(a.crypto||a.msCrypto).getRandomValues(e)),m(e)}catch(r){var t=a.navigator,n=t&&t.plugins;return[+new Date,a,n,a.screen,m(s)]}}():e,3),r),g=new h(r),y=function(){for(var e=g.g(6),t=u,n=0;e<l;)e=(e+n)*256,t*=256,n=g.g(1);for(;e>=p;)e/=2,t/=2,n>>>=1;return(e+n)/t};return y.int32=function(){return 0|g.g(4)},y.quick=function(){return g.g(4)/4294967296},y.double=y,f(m(g.S),s),(t.pass||n||function(e,t,n,r){return(r&&(r.S&&d(r,g),e.state=function(){return d(g,{})}),n)?(i.random=e,t):e})(y,c,"global"in t?t.global:this==i,t.state)}function h(e){var t,n=e.length,r=this,a=0,s=r.i=r.j=0,i=r.S=[];for(n||(e=[n++]);a<256;)i[a]=a++;for(a=0;a<256;a++)i[a]=i[s=255&s+e[a%n]+(t=i[a])],i[s]=t;(r.g=function(e){for(var t,n=0,a=r.i,s=r.j,i=r.S;e--;)t=i[a=255&a+1],n=256*n+i[255&(i[a]=i[s=255&s+t])+(i[s]=t)];return r.i=a,r.j=s,n})(256)}function d(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function f(e,t){for(var n,r=e+"",a=0;a<r.length;)t[255&a]=255&(n^=19*t[255&a])+r.charCodeAt(a++);return m(t)}function m(e){return String.fromCharCode.apply(0,e)}if(f(i.random(),s),e.exports){e.exports=c;try{o=n(5042)}catch(g){}}else void 0!==(r=(function(){return c}).call(t,n,t,e))&&(e.exports=r)}("undefined"!=typeof self?self:this,[],Math)}}]);