from typing import Iterable, Optional, Union

import torch

from .nodes import Nodes


class WaveLIFNodes(Nodes):
    # language=rst
    """
    Layer of Klein-Gordon wave neurons.

    Membrane potential follows second-order oscillatory dynamics driven by
    the Klein-Gordon dispersion relation instead of first-order exponential
    leaky integration.

    Update rule (leapfrog):

    .. math::

        v(t+1) = 2 v(t) - v_{prev}(t) + dt^2 (-\\chi^2 v(t) + x(t))

    where :math:`\\chi` is the natural oscillation frequency (mass term) and
    :math:`x` is the summed input from incoming connections.

    The neuron fires when ``v >= thresh`` and follows the standard
    refractory/reset protocol.

    Unlike LIF neurons that exponentially decay toward rest, WaveLIF neurons
    oscillate at frequency :math:`\\chi`, enabling frequency-selective
    resonance: inputs near :math:`\\omega = \\chi` are amplified while
    off-resonance inputs are attenuated.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = 1.0,
        reset: Union[float, torch.Tensor] = 0.0,
        refrac: Union[int, torch.Tensor] = 5,
        chi: Union[float, torch.Tensor] = 0.5,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Klein-Gordon wave neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param chi: Natural oscillation frequency (mass term).
            Controls the resonance frequency and dispersion.
            Must satisfy ``dt * chi < 2`` for stability.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
            **kwargs,
        )

        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )
        self.register_buffer(
            "chi", torch.tensor(chi, dtype=torch.float)
        )
        self.register_buffer("v", torch.FloatTensor())
        self.register_buffer("v_prev", torch.FloatTensor())
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )

        if lbound is None:
            self.lbound = None
        else:
            self.lbound = torch.tensor(lbound, dtype=torch.float)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Mask inputs during refractory period.
        x.masked_fill_(self.refrac_count > 0, 0.0)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Klein-Gordon leapfrog update:
        # v_new = 2*v - v_prev + dt^2 * (-chi^2 * v + x)
        dt2 = self.dt * self.dt
        v_new = (
            2.0 * self.v
            - self.v_prev
            + dt2 * (-self.chi * self.chi * self.v + x)
        )

        # Shift history.
        self.v_prev = self.v.clone()
        self.v = v_new

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.v_prev.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.reset)
        self.v_prev.fill_(self.reset)
        self.refrac_count.zero_()

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.  For WaveLIF the decay is implicit in the
        leapfrog update (no exponential decay constant).  This method only
        stores ``dt`` and validates the CFL stability condition.
        """
        super().compute_decays(dt=dt)
        if self.dt * self.chi.max().item() >= 2.0:
            raise ValueError(
                f"CFL instability: dt*chi = "
                f"{self.dt * self.chi.max().item():.3f} >= 2. "
                f"Reduce dt or chi."
            )

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.reset * torch.ones(
            batch_size, *self.shape, device=self.v.device
        )
        self.v_prev = self.reset * torch.ones(
            batch_size, *self.shape, device=self.v_prev.device
        )
        self.refrac_count = torch.zeros_like(
            self.v, device=self.refrac_count.device
        )


class DampedWaveLIFNodes(WaveLIFNodes):
    # language=rst
    """
    Layer of damped Klein-Gordon wave neurons.

    Adds a viscous damping term to the Klein-Gordon oscillator, bridging
    between the purely oscillatory :class:`WaveLIFNodes` and the
    exponentially-decaying ``LIFNodes``.

    Update rule (damped leapfrog):

    .. math::

        v(t+1) = \\frac{2 v(t) - (1 - \\gamma dt/2) v_{prev}(t)
        + dt^2 (-\\chi^2 v(t) + x(t))}{1 + \\gamma dt/2}

    where :math:`\\gamma` controls the damping rate.  At
    :math:`\\gamma = 0` this reduces to :class:`WaveLIFNodes`.
    As :math:`\\gamma \\to \\infty` the neuron approaches first-order decay.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = 1.0,
        reset: Union[float, torch.Tensor] = 0.0,
        refrac: Union[int, torch.Tensor] = 5,
        chi: Union[float, torch.Tensor] = 0.5,
        gamma: Union[float, torch.Tensor] = 0.1,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of damped Klein-Gordon wave neurons.

        :param gamma: Damping coefficient. 0 = undamped oscillator.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
            thresh=thresh,
            reset=reset,
            refrac=refrac,
            chi=chi,
            lbound=lbound,
            **kwargs,
        )

        self.register_buffer(
            "gamma", torch.tensor(gamma, dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step with damping.

        :param x: Inputs to the layer.
        """
        # Mask inputs during refractory period.
        x.masked_fill_(self.refrac_count > 0, 0.0)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Damped Klein-Gordon leapfrog update.
        dt = self.dt
        dt2 = dt * dt
        gd2 = self.gamma * dt * 0.5

        v_new = (
            2.0 * self.v
            - (1.0 - gd2) * self.v_prev
            + dt2 * (-self.chi * self.chi * self.v + x)
        ) / (1.0 + gd2)

        # Shift history.
        self.v_prev = self.v.clone()
        self.v = v_new

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.v_prev.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        # Call Nodes.forward (skip WaveLIFNodes.forward to avoid double update)
        Nodes.forward(self, x)
