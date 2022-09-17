<h1>Formula & Concept Sheet until Quiz 1</h1>

<h2>Concept</h2>

<h3>Lecture 1

Intensity, Gray Level, Image Sensor, Pixel, Coordinates, Sampling/Discretization, Quantization, Spatial Resolution, Gray-Level Resolution, Color Space, Color Primaries, Additive Primaries, Subtractive Primaries

<h3>Lecture 2

Visible Light, Rods and Cones, Spectral Sensitivity, Luminance, Brightness, Simultaneous Contrast, Weber’s Law, Lightness, Chrominance, Hue, Saturation, Chromatic Adaptation, CIE-XYZ, CIE-L\*a\*b*, Color Difference

<h3>Lecture 3

Impulse Sequence, Linear and Shift Invariant System, Principle of Superposition, Impulse Response, Convolution, Fourier Transform, Spatial/Frequency Variables, Basis Functions, Fourier/Power Spectrum, Discrete Fourier Transform, Conjugate Symmetry, Discrete Cosine Transform, Properties of Transform, Periodicity, Separable Property

<h3>Lecture4

Bandwidth, Band-Limited Signal Sampling, Function Sampling, Frequency/Rate, Sampling Interval, Dirac Delta Function, Sampling Theorem, Nyquist Frequency/Rate/Interval, Aliasing, Decision Level/Interval, Reconstruction Level, Quantization Error, Mean Square Error (MSE), Llyod-Max Quantizer, Linear Quantizer, Signal-to-Noise Ratio (SNR)

<h3>Lecture5</h3>

Image Enhancement, Point Processing, Gray-Level Reversal, Low/High Contrast, Contrast Stretching, Thresholding, Dynamic Range, Image Histogram, Histogram Equalization, Histogram Matching, Image Filtering, Lowpass/Highpass Filters, Spatial Filtering, Smoothing/Sharpening Filters, Unsharp Masking, High-Boost Filters, Gradient/Edge Filters



<h2>Formula

<h3>Lecture 1

- Color space 

$$
Color \ space: \ C = aP_1 + bP_2 + CP_3<h3>Lecture 2
$$

- Luminance (CIE definition)
  $$
  Y = \int_\lambda R(\lambda)V(\lambda)d\lambda, \ where \ R(\lambda)=\rho(\lambda)E(\lambda) \\
  \rho: \ refelctivity \ or \ transmissivity; \  E: \ energy \ distribution \ at \ \lambda; \ \lambda: \ wavelength
  $$

- Simultaneous Contrast - *Weber's Law*
  $$
  \frac{|Y_s-Y|}{Y} \approx d(lnY) = \Delta C = constant
  $$

- Lightness (w.r.t *luminance*)
  $$
  L^* = \begin{cases}
  116(\frac{Y}{Y_n})^{\frac{1}{3}}-16, \ \ \ if \frac{Y}{Y_n} > 0.008856 \\
  903.3\frac{Y}{Y_n}, \ \ \ otherwise
  \end{cases}
  $$

- RGB - XYZ Conversion
  $$
  \begin{bmatrix}X\\Y\\Z\end{bmatrix}=\begin{bmatrix}0.4124 &0.3576 &0.1804\\0.2127 &0.7152 &0.0722\\0.0193 &0.1192 &0.9502\end{bmatrix}  \begin{bmatrix}R_{709}\\G_{709}\\B_{709}\end{bmatrix}; \ \ 
  
  \begin{bmatrix}R_{709}\\G_{709}\\B_{709}\end{bmatrix}=\begin{bmatrix}3.2405 &-1.5372 &-0.4985\\-0.9692& 1.8760& 0.0416\\0.0556& -0.2040 &1.0573\end{bmatrix}  \begin{bmatrix}X\\Y\\Z\end{bmatrix}
  $$

- XYZ - Lab Conversion

$$
L^*=116f(\frac{Y}{Y_n})-16, \\ 
a^*=500(f(\frac{X}{X_n})-f(\frac{Y}{Y_n}), \\ 
b^*=200(f(\frac{Y}{Y_n})-f(\frac{Z}{Z_n}), \\

where \ \ f(u) = 
\begin{cases}
u^{\frac{1}{3}} \ \ \ if u > 0.008856 \\
7.787u+\frac{16}{116} \ \ \ otherwise
\end{cases} \\

Chroma: \ C^*_{ab} = \sqrt{(a^*)^2+(b^*)^2}; \\ Hue: \ h_{ab} = tan^{-1}(\frac{b^*}{a^*}); \\  Color \ difference: \ \Delta E = \sqrt{(\Delta L^*)^2+(\Delta a^*)^2+(\Delta b^*)^2}
$$

<h3>Lecture 3</h3>

* Definition of image as 2D sequence
  $$
  f(x,y) = \sum^{\infin}_{m=-\infin} \sum^{\infin}_{n=-\infin}f(m,n)\delta(x-m,y-n)
  $$

* 2D LSI (Linear Shift-Invariant) System
  $$
  g(x,y) = T[f(x,y)] = \sum^{\infin}_{m=-\infin} \sum^{\infin}_{n=-\infin}f(m,n) \ T[\delta(x-m,y-n)] \\
  =\sum^{\infin}_{m=-\infin} \sum^{\infin}_{n=-\infin}f(m,n) h(x-m,y-n) =f(x,y)*h(x,y)
  $$

* Fourier Transform
  $$
  F(u,v)=\int_{-\infin}^{\infin} f(x,y) \ e^{-j2\pi(ux+vy)} \ dxdy \\
  f(x,y)=\int_{-\infin}^{\infin} f(u,v) \ e^{j2\pi(ux+vy)} \ dxdy \\\\
  Fourier \ spectrum: |F(u,v)|=\sqrt{R^2(u,v)+I^2(u,v)} \\
  Phase \ angle:\phi = tan^{-1}[\frac{I(u,v)}{R(u,v)}] \\
  Power \ spectrum = |F(u,v)|^2 = R^2(u,v)+I^2(u,v), \ \ \ where F(u,v) = R(u,v)+jI(u,v)
  $$

* Discrete Fourier Transform
  $$
  F(u,v)=\frac{1}{MN}\sum_{m=0}^{\infin}\sum_{n=0}^{\infin}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})} \\
  f(x,y)=\sum_{m=0}^{\infin}\sum_{n=0}^{\infin}F(u,v)e^{j2\pi(\frac{ux}{M}+\frac{vy}{N})}
  $$

* Properties of 2D FT
  $$
  f(x,y)e^{j2\pi(\frac{u_0x}{M})+\frac{v_0y}{N})} \Leftrightarrow F(u-u_0, v-v_0) \\
  f(x-x_0,y-y_0) \Leftrightarrow F(u, v)e^{-j2\pi(\frac{u_0x}{M})+\frac{v_0y}{N})} \\
  f(ax,by)\Leftrightarrow \frac{1}{|ab|}f(\frac{u}{a},\frac{v}{b}) \\
  f*h \Leftrightarrow FH, \ fh \Leftrightarrow F*H
  $$

* DCT (Discrete Cosine Transform)
  $$
  F(u,v) = \frac{2}{N}C(u)C(v)\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)cos(\frac{(2x+1)u\pi}{2N})cos(\frac{(2y+1)v\pi}{2N}),\\ 
  f(x,y) = \frac{2}{N}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}C(u)C(v)F(u,v)cos(\frac{(2x+1)u\pi}{2N})cos(\frac{(2y+1)v\pi}{2N}), \\
  where \ \ C(u), C(v)= \begin{cases}\frac{1}{\sqrt{2}} \ for \ u,v=0\\1 \ for \ u,v=1,...,N-1\end{cases}
  $$

<h3>Lecture4

* Bandwidth: F(u,v)=0 for any |u|, |v| greater than U_0, V_0

* Image Sampling
  $$
  In \ spatial \ domain: \ f_s(x,y) = F(x,y)x(x,y)=\sum_{m=-\infin}^{\infin}\sum_{n=-\infin}^{\infin}f(m\Delta x,n\Delta y)\delta(x-m\Delta x,y-n\Delta y) \\
  
  In \ spectral \ domain: \ F_s(u,v) = F(u,v)*S(u,v)=\frac{1}{\Delta x\Delta y}\sum_{k=-\infin}^{\infin}\sum_{l=-\infin}^{\infin}F(u,v)*\delta(u-\frac{k}{\Delta x},v-\frac{l}{\Delta y})\\
  =\frac{1}{\Delta x\Delta y}\sum_{k=-\infin}^{\infin}\sum_{l=-\infin}^{\infin}F(u-\frac{k}{\Delta x},v-\frac{l}{\Delta y})
  $$

* Sampling Theorem - *Nyquist Rate*
  $$
  f_{xs}=\frac{1}{\Delta x}>2U_0 \ \ f_{ys}=\frac{1}{\Delta y}>2V_0 \\
  $$
  Reconstructing original image from sampled image
  $$
  \tilde{f}(x,y) = \sum_{m=-\infin}^{\infin}\sum_{n=-\infin}^{\infin} f(m\Delta x, n\Delta y) sinc(xf_{xs}-m)sinc(yf_{ys}-n)
  $$

* Definition of Quantization
  $$
  \bar{f} = Q(f) = r_k, \ if \ \ t_k\leq f\leq t_{k+1} \ \ for \ k = 1,...,L
  $$
  Lloyd-Max Quantizer - based on *probability density function*

$$
Quantization \ error: \ \varepsilon=E[(f-\bar{f})^2]=\int_{t_1}^{t_{L+1}}(f-Q(f))^2 P(f) df \\
=\sum_{k=1}^{L}\int_{t_k}^{t_{k+1}}(f-Q(f))^2 df \\
Let \ \frac{d\varepsilon}{df}=0, \ we \ get: \\
t_k = \frac{r_k+r_{k-1}}{2}, \ for \ k=2,...,L \\
r_k = \frac{\int_{t_k}^{t_{k+1}}f P_f(f) df}{\int_{t_k}^{t_{k+1}} P_f(f) df}, \ for \ k=1,...,L
$$

​		If p(f) is uniform (namely, ):
$$
p(f) = \frac{1}{t_{k+1}-t_1}\\
t_k = t_{k-1} + q, \ r_k = t_k+\frac{q}{2}, where \ q \ is \ defined \ as \ q=\frac{t_{L+1}-t_1}{L} \\
In \ this \ case: \ \varepsilon_{linear} = \frac{q^2}{12}, and \ SNR=10log_{10}\frac{\sigma^2}{\varepsilon} = 10log_{10}2^{2B} \approx 6B \ dB
$$

<h3>Lecture5

- Image Histogram
  $$
  histogram: \ \sum_{k=0}^{L-1}h_f(k)=MN, \ where \ L \ is \ number \ of gray \ level, \ MN  \ is \ pixels \ amount
  $$

- Histogram equalization - to get a *uniform* histogram
  $$
  probabilities: \ p_f(k) = \frac{h_f(k)}{\sum_{k=0}^{L-1}h_f(k)} for \ k=0,...,L-1 \\
  pdf: \ c_f(f)=\sum_{k=0}^{f}p_f(k), for \ f=0,...,L-1 \\
  Thus, \ equalized \ histogram: \ g=T(f)=Round[(\frac{c_f(f)-c_{min}}{1-c_{min}})(L-1)]
  $$
  

* Filtering - **to be continued**