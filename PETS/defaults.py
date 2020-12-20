from dotmap import DotMap
from PETS.dynamics_model import nn_constructor
from PETS.env.cartpole import CartpoleEnv
from PETS.config.cartpole import CartpoleConfigModule

params = DotMap(
    env=None,
    prop_cfg=DotMap(
        model_init_cfg=DotMap(
            num_nets=3,
            input_dim=None,
            output_dim=None,
            model_constructor=nn_constructor,
        ),
        model_train_cfg=DotMap(
            epochs=5,
        ),
        obs_preproc=None,
        obs_postproc=None,
        targ_proc=None,
        model_pretrained=False,
        npart=12,
        ign_var=False,
        mode="TSinf",
    ),
    opt_cfg=DotMap(
        mode="CEM",
        plan_hor=20,
        obs_cost_fn=None,
        ac_cost_fn=None,
        cfg=DotMap(
            alpha=0.2,
            max_iters=3,
            num_elites=2,
            popsize=5,
        ),
    ),
)


def get_cartpole_defaults():
    defaults = params.copy()
    defaults.env = CartpoleEnv()
    defaults.prop_cfg.model_init_cfg.input_dim = 6
    defaults.prop_cfg.model_init_cfg.output_dim = 4
    defaults.prop_cfg.obs_preproc = CartpoleConfigModule.obs_preproc
    defaults.prop_cfg.obs_postproc = CartpoleConfigModule.obs_postproc
    defaults.prop_cfg.targ_proc = CartpoleConfigModule.targ_proc
    defaults.opt_cfg.obs_cost_fn = CartpoleConfigModule.obs_cost_fn
    defaults.opt_cfg.ac_cost_fn = CartpoleConfigModule.ac_cost_fn
    return defaults
