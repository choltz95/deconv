## Brief summary of the deconvolution problem

---

A brief summary of some techniques for denoising and deconvolving calcium imaging data. Below, we review the background provided by Vogelstein et. al., 2009.

We assume that the observed fluorescence trace $y_t$  is a noisy version of the unobserved calcium concentration $c_t$. Furthermore, we assume that the calcium concentration decays exponentially, unless there is a spike, in which case the calcium concentration increases instantaneously. More precisely,
$$
y_t = c_t + \epsilon_t, \epsilon \sim (0,\sigma^2), t=1,\ldots, T \\
c_t = \gamma c_{t-1} + z_t, t=1,\ldots, T
$$
where $z_t \geq 0$ indicates the presence of a spike at the $t$-th timestep, and $Z$ is modeled as a poisson random variable with constant spike rate.
$$
z_t \stackrel{iid}{\sim}Poisson(\lambda)
$$
Given the above model, the goal is to find the maximum a posteriori spike train, i.e., the most likely spike train, $\hat{n}$, given the fluorescence measurements, $\mathbf{F}$:
$$
\hat{z} = \text{argmax}P(z | y) = \text{argmax}\frac{1}{P(y)}P(y | z)P(z)
$$
and we have that $P(z | y) = \mathcal{N}(c + \epsilon, \sigma^2)$, $P(z) = Poisson(\lambda)$. Following through, we get that
$$
\hat{z} = \text{argmax} \sum_{t=1}^T-\frac{1}{2\sigma^2}(y_t - c_t - \epsilon_t)^2 + z_t\lambda
$$

### Discrete Binning

----

Discrete binning is implemented as a high-pass filter followed by max-normalized discrete binning.

### Wiener Process

____

The Wiener filter is performs optimal linear deconvolution by implicitly approximating the Poisson spike rate with a Gaussian spike rate. Following the appendix from Vogelstein et. al., 2010:

The Poisson distribution characterizing $z_t$ can be approximated by an appropriately parameterized Gaussian.
$$
z_t \stackrel{iid}{\sim}\mathcal{N}(\lambda, \lambda)
$$
The associated MAP program yields the following program:
$$
\hat{z_t} = \text{argmax}_{z_t}\sum_{t=1}^T(\frac{1}{2\sigma^2}y_t - \alpha c_t - \beta)^2 + \frac{1}{\lambda} (z_t - \lambda)^2
$$
Note that this solution is the optimal linear solution, under the assumption that spikes follow a Gaussian distribution, and is equivalent to the Wiener filter.

### FOOPSI (Fast Optimal OPtical Spike Inference )

-----

A general form of the FOOPSI problem
$$
\text{minimize}_{\mathbf{c}, \mathbf{s}} ~ \frac{1}{2} \|\mathbf{y}-\mathbf{c}\|_2^2 + \lambda \|\mathbf{s}\|_1 
~
\text{subject to } c_t = \sum_i^p \gamma_{i=1} c_{t-i}+s_t \text{ with } s_t=0 \text{ or } s_t\geq s_{min}
$$
$\mathbf{y}$ is the noisy raw calcium trace and $\mathbf{c}$ is the desired denoised trace. Typically, $\lambda$ and $\gamma$ are known or provided. We use an auto-regressive (AR) $c_t = \sum_i^p \gamma_{i=1} c_{t-i}+s_t$ model to model $\mathbf{c}$ and the corresponding spiking activity $\mathbf{s}$. A more general model of $\mathbf{c}$ and $\mathbf{s}$ is $\mathbf{c} = \mathbf{s} \ast \mathbf{h}$, where $\ast$ is an arbitrary convolution operation. The convolution kernel $\mathbf{h}$ can be interpreted as calcium responses of a neuron after firing a spike.

The AR model is equivalent to special kernel $\mathbf{h}$. For example, an AR-1 model ($p=1$) corresponds to
$$
h(t) = \exp(-t/\tau)
$$
and AR-2 ($p=2$) model correspond to
$$
h(t) = \exp(-t/\tau_d) - \exp(-t/\tau_r).
$$

----

#### Constrained FOOPSI

$$
\text{minimize}_{\mathbf{c}, \mathbf{s}} ~ \|\mathbf{s}\|_1 
~
\text{subject to } c_t = \sum_i \gamma_{i=1}^p c_{t-i}+s_t \text{ and } \|\mathbf{y}-\mathbf{c}\|_2^2 = \sigma^2T
$$

This formulation is equivalent to the basic FOOPSI problem, but sets $\lambda$ automatically by restricting the residual sum of squares (RSS) to be $\sigma^2T$, where $\sigma$ is a noise level and can be estimated from the raw data.

----

#### Thresholded FOOPSI

----

Both methods described above adoopt an L-1 penalty to enforce the sparsity of the spiking signal $\mathbf{s}$. However, the estimated $\mathbf{s}$ usually contains many false positive spikes with small amplitudes to significant detriment to the overall estimation. 

Thresholded-FOOPSI integrates a thresholding step into the deconvolution framework by setting a hard threshold of spike size $s_{min}$. $s_{min}$ is typically chosen to be $3\sigma$, where $\sigma$ is estimated from the raw data.
$$
\text{minimize}_{\mathbf{c}, \mathbf{s}} ~ \frac{1}{2} \|\mathbf{y}-\mathbf{c}\|_2^2 
~
\text{subject to } c_t = \sum_i \gamma_{i=1}^p c_{t-i}+s_t \text{ with } s_t=0 \text{ or } s_t\geq s_{min}
$$