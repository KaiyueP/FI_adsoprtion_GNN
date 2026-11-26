"""Equivariant radial geometry model with an explicit initial-force correction.

Opt-in architecture: legacy SEMPNN checkpoints retain their original class.
Force scalars are rotation/reflection invariants; full polar vectors are retained.
The cutoff smoothly damps messages, not the full hard-truncated graph topology.
"""
import math
import torch
from torch import nn
from e3nn.o3 import Irreps
from .SEMPNN import SEMPNN
from .Layers.SEMPLayer import SEMPLayer
from .BalancedIrreps import BalancedIrreps
from .PooledGraphNN import adsorbate_mean_pool
from Dataset.adsorption_records import force_mode


class RadialMessageLayer(SEMPLayer):
    """Scalar invariant gates act equally on every component of each irrep."""
    def __init__(self, *args, radial_width=32, **kwargs):
        super().__init__(*args, **kwargs)
        count=kwargs['additional_message_irreps'].dim
        self.gates=nn.Sequential(nn.Linear(count-1,radial_width),nn.SiLU(),
                                 nn.Linear(radial_width,self.hidden_irreps.num_irreps))
        # Index one scalar gate per irrep copy, never per Cartesian component.
        indices=[];gate=0
        for multiplicity,irrep in self.hidden_irreps:
            for _ in range(multiplicity):
                indices.extend([gate]*irrep.dim);gate+=1
        self.register_buffer('gate_index',torch.tensor(indices,dtype=torch.long),persistent=False)

    def message(self,x_i,x_j,edge_attr,additional_message_features):
        message=super().message(x_i,x_j,edge_attr,additional_message_features)
        gates=2*torch.sigmoid(self.gates(additional_message_features[:,:-1]))
        envelope=additional_message_features[:,-1:]
        return message*gates[:,self.gate_index]*envelope/math.sqrt(24.)


class RadialSEMPNN(SEMPNN):
    def __init__(self,*args,radial_width=32,**kwargs):
        self.radial_width=radial_width
        super().__init__(*args,**kwargs)

    def _make_layer(self):
        return RadialMessageLayer(self.hidden_irreps,self.hidden_irreps,self.hidden_irreps,
            self.edge_attr_irreps,self.node_attr_irreps,norm=self.norm,
            additional_message_irreps=self.additional_message_irreps,radial_width=self.radial_width)


class ForceResidualGNN(nn.Module):
    """E = geometry + g(|F|) * correction(geometry, F).

    The force correction output is zero-initialized. The magnitude gate ensures
    exactly zero correction when all supplied force vectors are zero, including
    after training. This is a prediction model, not a conservative force field.
    """
    def __init__(self,config,feature_count):
        super().__init__()
        if config.get('z_symmetry_breaking',False):
            raise ValueError('force_residual_v1 uses no fixed Cartesian reference vector.')
        self.use_forces=force_mode(config)!='none'
        self.scalar_count=feature_count-3*int(self.use_forces)
        self.radius=float(config['radius'])
        radial_count=config.get('radial_basis_count',16)
        radial_width=config.get('radial_hidden_features',32)
        if type(radial_count)is not int or radial_count<2:
            raise ValueError('radial_basis_count must be an integer >= 2.')
        if type(radial_width)is not int or radial_width<1:
            raise ValueError('radial_hidden_features must be positive.')
        if not math.isfinite(self.radius) or self.radius<=0:
            raise ValueError('radius must be finite and positive.')
        self.register_buffer('radial_centers',torch.linspace(0,self.radius,radial_count))
        self.radial_width_A=self.radius/(radial_count-1)
        angular=Irreps.spherical_harmonics(config['lmax_h'])
        hidden=BalancedIrreps(config['lmax_h'],config['hidden_features'])
        common=dict(hidden_irreps=hidden,output_irreps_lst=[Irreps('1x0e')],
            edge_attr_irreps=angular,node_attr_irreps=angular,
            num_graph_layers=config['num_graph_layers'],num_prepool_layers=config['num_prepool_layers'],
            num_postpool_layers=config['num_postpool_layers'],norm=config['layer_norm'],
            pooling=config['pooling'],radial_width=radial_width)
        # Construct geometry first: same seed produces the same initial geometry
        # model whether force inputs are enabled or disabled.
        self.geometry=RadialSEMPNN(input_irreps=Irreps(f'{self.scalar_count}x0e'),
            additional_message_irreps=Irreps(f'{radial_count+1}x0e'),**common)
        if self.use_forces:
            self.force_correction=RadialSEMPNN(
                input_irreps=Irreps(f'{self.scalar_count+1}x0e+1x1o'),
                additional_message_irreps=Irreps(f'{radial_count+6}x0e'),**common)
            final=self.force_correction.mlp.outputs[0][-1]
            for parameter in final.parameters():nn.init.zeros_(parameter)

    def edge_features(self,graph,forces=None):
        distances=graph.edge_dis
        ratio=(distances/self.radius).clamp(0,1)
        envelope=.5*(torch.cos(math.pi*ratio)+1)
        envelope=torch.where(distances<self.radius,envelope,torch.zeros_like(envelope))
        radial=torch.exp(-.5*((distances-self.radial_centers)/self.radial_width_A)**2)
        radial=radial*envelope
        if forces is None:return torch.cat([radial,envelope],dim=-1)
        source,target=graph.edge_index
        # Match the force representation's [y,z,x] model coordinate frame.
        unit=graph.r_ij[:,[1,2,0]]/distances.clamp_min(1e-8)
        fi,fj=forces[source],forces[target]
        ni,nj=fi.norm(dim=-1,keepdim=True),fj.norm(dim=-1,keepdim=True)
        cosine=(fi*fj).sum(dim=-1,keepdim=True)/(ni*nj).clamp_min(1e-8)
        invariants=torch.cat([torch.asinh((fi*unit).sum(dim=-1,keepdim=True)),
            torch.asinh((fj*unit).sum(dim=-1,keepdim=True)),torch.log1p(ni),torch.log1p(nj),
            cosine.clamp(-1,1)],dim=-1)
        return torch.cat([radial,invariants,envelope],dim=-1)

    def forward_components(self,graph):
        geometric=graph.clone()
        geometric.x=graph.x[:,:self.scalar_count]
        geometric.edge_dis=self.edge_features(graph)
        geometry=self.geometry(geometric)[0]
        correction=torch.zeros_like(geometry)
        gate=torch.zeros_like(geometry)
        if self.use_forces:
            forces=graph.x[:,self.scalar_count:self.scalar_count+3]
            magnitude=torch.log1p(forces.norm(dim=-1,keepdim=True))
            informed=graph.clone()
            informed.x=torch.cat([geometric.x,magnitude,forces],dim=-1)
            informed.edge_dis=self.edge_features(graph,forces)
            raw=self.force_correction(informed)[0]
            mean=adsorbate_mean_pool(magnitude,graph.batch,graph.adsorbate_mask)
            gate=mean/(1+mean)
            correction=gate*raw
        return {'geometry':geometry,'force_correction':correction,'force_gate':gate,
                'total':geometry+correction}

    def forward(self,graph):
        return [self.forward_components(graph)['total']]
