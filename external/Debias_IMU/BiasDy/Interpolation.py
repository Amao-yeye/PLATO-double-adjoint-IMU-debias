"""Cubic Hermite IMU interpolation used by the native-grid rollout."""
import torch


def get_cubic_hermite_coeffs(time: torch.Tensor, data: torch.Tensor)->torch.Tensor:
    """ time: torch.Tensor of shape (n,); data: torch.Tensor of shape (...,n, d) """
    """return a, b, c, d of shape (..., n, d) , where a + b * t + c * t^2 + d * t^3 = data (1-d case see CubicHermiteSpline class for more details)"""
    deriv = torch.zeros_like(data)
    deriv[..., 1:, :] = (data[..., 1:, :] - data[..., :-1, :]) / (time[1:] - time[:-1]).unsqueeze(-1) # backward difference

    # hermite cubic spline
    time_tmp = (time[1:] - time[:-1]).unsqueeze(-1)
    a = data[..., :-1, :]
    # b = deriv[..., :-1, :]
    # time_tmp_2 = time_tmp * time_tmp
    # time_tmp_3 = time_tmp * time_tmp_2
    # c = 3 * (data[..., 1:, :] - data[..., :-1, :]) / time_tmp_2 - (2 * deriv[..., :-1, :] + deriv[..., 1:, :]) / time_tmp
    # d = 2 * (data[..., :-1, :] - data[..., 1:, :]) / time_tmp_3 + (deriv[..., :-1, :] + deriv[..., 1:, :]) / time_tmp_2
    b = deriv[..., :-1, :] * time_tmp
    c = 3 * (data[..., 1:, :] - data[..., :-1, :]) - (2 * deriv[..., :-1, :] + deriv[..., 1:, :]) * time_tmp
    d = 2 * (data[..., :-1, :] - data[..., 1:, :]) + (deriv[..., :-1, :] + deriv[..., 1:, :]) * time_tmp
    return a, b, c, d


class CubicHermiteSpline:
    """Calculates the hermite cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        times = torch.linspace(0, 1, 7)
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        X = torch.rand(2, 1, 7, 3)
        coeffs = get_cubic_hermite_coeffs(times, X)
        # ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...
        spline = NaturalCubicSpline(times, coeffs)
        t = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(t)
    """

    def __init__(self, times, coeffs, device = 'cpu', **kwargs):
        """
        Arguments:
            times: As was passed as an argument to natural_cubic_spline_coeffs.
            coeffs: As returned by natural_cubic_spline_coeffs.
        """
        super().__init__(**kwargs)

        a, b, c, d = coeffs

        # ``torch.searchsorted`` below requires a sorted one-dimensional
        # boundary tensor. Keeping it contiguous avoids an implicit copy (and
        # the associated warning) on every spline query.
        self._times = times.to(device).contiguous()
        self._a = a.to(device)
        self._b = b.to(device)
        self._c = c.to(device)
        self._d = d.to(device)
        self._device = device

    def check_t_dim(self, t):
        if t.dim() == 0:
            return True
        elif t.dim() == 1:
            return False
        else:
            raise ValueError("t must be a scalar or a 1D tensor of size 1.")

    def _interpret_t(self, t: torch.Tensor):
        single_t = self.check_t_dim(t)
        if single_t:
            # assert t >= self._times[0], "t must be greater than or equal to the first time in the spline"
            maxlen = self._b.size(-2) - 1
            # Preserve the legacy scalar convention exactly: choose the
            # interval strictly to the left of an interior knot.
            index = torch.searchsorted(self._times, t, right=False) - 1
            index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
            # will never access the last element of self._times; this is correct behaviour
            fractional_part = t - self._times[index]
        else: # batched
            maxlen = self._b.size(-2) - 1
            # This is equivalent to
            # ``(t.unsqueeze(-1) >= self._times).sum(-1) - 1`` but allocates
            # only O(batch) storage instead of O(batch * sequence_length).
            # ``right=True`` preserves the legacy batched convention at knots.
            index = torch.searchsorted(
                self._times,
                t.contiguous(),
                right=True,
            ) - 1
            index = index.clamp(0, maxlen)
            fractional_part = t - self._times[index] # (batch_size,)
        return fractional_part, index

    def evaluate(self, t):
        single_t = self.check_t_dim(t)
        if single_t:
            """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
            fractional_part, index = self._interpret_t(t)
            fractional_part = fractional_part / (self._times[index + 1] - self._times[index])
            inner = self._c[..., index, :] + self._d[..., index, :] * fractional_part
            inner = self._b[..., index, :] + inner * fractional_part
            return self._a[..., index, :] + inner * fractional_part
        else:
            fractional_part, index = self._interpret_t(t)
            fractional_part = fractional_part / (self._times[index + 1] - self._times[index])
            fractional_part = fractional_part.unsqueeze(-1)
            if self._a[..., index, :].dim() == 3:
                fractional_part = fractional_part.unsqueeze(0)
            fractional_part = fractional_part.expand(*self._a[..., index, :].shape)
            inner = self._c[..., index, :] + self._d[..., index, :] * fractional_part
            inner = self._b[..., index, :] + inner * fractional_part
            return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        single_t = self.check_t_dim(t)
        if single_t:
            """Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
            fractional_part, index = self._interpret_t(t)
            fractional_part = fractional_part / (self._times[index + 1] - self._times[index])
            inner = 3 * self._d[..., index, :] * fractional_part + self._c[..., index, :] * 2
            return (self._b[..., index, :] + inner * fractional_part) / (self._times[index + 1] - self._times[index])
        else:
            fractional_part, index = self._interpret_t(t)
            tmp = (self._times[index + 1] - self._times[index])
            fractional_part = fractional_part / tmp
            fractional_part = fractional_part.unsqueeze(-1)
            tmp = tmp.unsqueeze(-1)
            if self._a[..., index, :].dim() == 3:
                fractional_part = fractional_part.unsqueeze(0)
                tmp = tmp.unsqueeze(0)
            fractional_part = fractional_part.expand(*self._a[..., index, :].shape)
            inner = 3 * self._d[..., index, :] * fractional_part + self._c[..., index, :] * 2
            return (self._b[..., index, :] + inner * fractional_part) / tmp
