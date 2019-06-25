import os
import sys

if sys.hexversion >= 0x03000000:

    def load_model(model_file='SMPLH_female.pkl', ncomps=12):
        return _SmplModelClient(model_file, ncomps)
else:

    def load_model(model_file='SMPLH_female.pkl', ncomps=12):
        return _SmplModelDirect(model_file, ncomps)


class _SmplModelDirect():
    """ Use this class direct from python 2.7 """

    def __init__(self, model_file, ncomps):
        self._model = _load_model(model_file, ncomps)

    @property
    def pose(self):
        return self._model.pose

    @property
    def betas(self):
        return self._model.betas

    @property
    def trans(self):
        return self._model.trans

    def generate_mesh(self, pose=None, betas=None, trans=None):
        if betas is not None:
            self._model.betas[:] = betas
        if pose is not None:
            self._model.pose[:] = pose
        if trans is not None:
            self._model.trans[:] = trans
        return self._model.r, self._model.f

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class _SmplModelClient():
    """ Use this wrapper from python 3.5 """

    def __init__(self, model_file, ncomps):
        self._model_file = model_file
        self._ncomps = ncomps

    @property
    def pose(self):
        return self._pose

    @property
    def betas(self):
        return self._betas

    @property
    def trans(self):
        return self._trans

    @property
    def J_transformed(self):
        return self._J_transformed

    def generate_mesh(self, pose=None, betas=None, trans=None,
                      center_idx=None):
        if pose is not None:
            self.pose[:] = pose
        if betas is not None:
            self.betas[:] = betas
        if trans is not None:
            self.trans[:] = trans
        if center_idx is not None:
            offset = self.J_transformed[center_idx, :]
            self.trans[:] = -offset
        self._proc.stdin.write(b'\n')
        self._proc.stdin.flush()
        self._proc.stdout.readline()
        return self._verts, self._faces

    def __enter__(self):
        args = [
            'python2',
            os.path.abspath(__file__), '--model_file', self._model_file,
            '--ncomps',
            str(self._ncomps)
        ]
        from subprocess import Popen, PIPE
        self._proc = Popen(args, stdin=PIPE, stdout=PIPE)
        assert self._proc.stdout.readline() == b'READY\n'

        import SharedArray as sa
        self._pose = sa.attach('shm://pose')
        self._betas = sa.attach('shm://betas')
        self._trans = sa.attach('shm://trans')
        self._verts = sa.attach('shm://verts')
        self._faces = sa.attach('shm://faces')
        self._J_transformed = sa.attach('shm://J_transformed')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            outs, errs = self._proc.communicate(input=b'DONE\n', timeout=2)
        except Exception:
            self._proc.kill()


def _load_model(model_file, ncomps):
    mano_path = os.environ.get('MANO_LOCATION', '~/mano')
    sys.path.append(mano_path)
    from webuser.smpl_handpca_wrapper import load_model
    if not os.path.isabs(model_file):
        model_file = os.path.join(mano_path, 'models', model_file)
    return load_model(model_file, ncomps=ncomps, flat_hand_mean=False)


def _run_server(model_file, ncomps):
    model = _load_model(model_file, ncomps)
    try:
        import SharedArray as sa
        for name in [
                'verts', 'faces', 'pose', 'betas', 'trans', 'J_transformed'
        ]:
            try:
                sa.delete(name)
            except:
                pass

        pose = sa.create('shm://pose', model.pose.shape, model.pose.dtype)
        betas = sa.create('shm://betas', model.betas.shape, model.betas.dtype)
        trans = sa.create('shm://trans', model.trans.shape, model.trans.dtype)
        verts = sa.create('shm://verts', model.r.shape, model.r.dtype)
        faces = sa.create('shm://faces', model.f.shape, model.f.dtype)
        J_transformed = sa.create('shm://J_transformed',
                                  model.J_transformed.shape,
                                  model.J_transformed.dtype)

        pose[:] = model.pose
        betas[:] = model.betas
        trans[:] = [0, 0,
                    0]  # model.J_transformed[25, :] + [0,0,-3] # model.trans
        faces[:] = model.f
        J_transformed[:] = model.J_transformed

        sys.stdout.write('READY\n')
        sys.stdout.flush()

        while True:
            cmd = sys.stdin.readline()
            if cmd == b'DONE\n':
                break
            model.betas[:] = betas
            model.pose[:] = pose
            model.trans[:] = trans
            # TODO see with Igor
            # model.J_transformed[:] = J_transformed
            verts[:] = model.r
            sys.stdout.write('\n')
            sys.stdout.flush()

    finally:
        sa.delete('pose')
        sa.delete('betas')
        sa.delete('trans')
        sa.delete('verts')
        sa.delete('faces')
        sa.delete('J_transformed')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Server.')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--ncomps', type=int)
    args = parser.parse_args()

    _run_server(args.model_file, args.ncomps)
