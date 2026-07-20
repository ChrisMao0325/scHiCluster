import os
import pathlib
import subprocess

import pandas as pd
import inspect

import schicluster
from .loop_calling import filter_loops, call_loops
from .merge_raw_matrix import make_raw_matrix_cell_table
from .shuffle_fdr import compute_t, permute_fdr, update_fdr_qval
from .merge_cell_to_group import merge_group_to_bigger_group_cools

PACKAGE_DIR = pathlib.Path(schicluster.__path__[0])

with open(PACKAGE_DIR / 'loop/generate_matrix_chunk.Snakefile') as tmp:
    GENERATE_MATRIX_CHUNK_TEMPLATE = tmp.read()
with open(PACKAGE_DIR / 'loop/generate_matrix_group.Snakefile') as tmp:
    GENERATE_MATRIX_SCOOL_TEMPLATE = tmp.read()


def prepare_dir(output_dir, chunk_df, dist, cap, pad, gap, resolution,
                min_cutoff, chrom_size_path, keep_cell_matrix, log_e_str, shuffle):
    output_dir.mkdir(exist_ok=True)
    cell_table_path = str((output_dir / 'cell_table.csv').absolute())
    chunk_df[['cell_url']].to_csv(cell_table_path, header=False, index=True)
    if shuffle:
        shuffle_str = '--shuffle'
    else:
        shuffle_str = ''
    parameters = dict(dist=dist,
                      cap=cap,
                      pad=pad,
                      gap=gap,
                      resolution=resolution,
                      min_cutoff=min_cutoff,
                      cell_table_path=f'"{cell_table_path}"',
                      chrom_size_path=f'"{chrom_size_path}"',
                      keep_cell_matrix=keep_cell_matrix,
                      log_e_str=f'"{log_e_str}"',
                      shuffle=shuffle,
                      shuffle_str=f'"{shuffle_str}"')
    parameters_str = '\n'.join(f'{k} = {v}'
                               for k, v in parameters.items())

    with open(output_dir / 'Snakefile', 'w') as f:
        f.write(parameters_str + GENERATE_MATRIX_CHUNK_TEMPLATE)
    return


def prepare_loop_snakemake(cell_table_path,
                           output_dir,
                           chrom_size_path,
                           chunk_size=100,
                           dist=5050000,
                           cap=5,
                           pad=5,
                           gap=2,
                           resolution=10000,
                           min_cutoff=1e-6,
                           keep_cell_matrix=False,
                           cpu_per_job=10,
                           log_e=True,
                           shuffle=False,
                           raw_resolution_str=None,
                           downsample_shuffle=None,
                           e_positive_only=True,
                           use_bkfilter=True):
    _cell_table_path = str(cell_table_path)
    sep = '\t' if _cell_table_path.endswith('tsv') else ','
    cell_table = pd.read_csv(cell_table_path, index_col=0, sep=sep, header=None,
                             names=['cell_id', 'cell_url', 'cell_group'])
    if shuffle and (downsample_shuffle is not None):
        # for shuffle background, downsample to downsample_shuffle to save time
        if cell_table.shape[0] > downsample_shuffle:
            cell_table = cell_table.sample(downsample_shuffle)
    output_dir = pathlib.Path(output_dir).absolute()
    output_dir.mkdir(exist_ok=True)

    # a single dir for raw matrix
    if raw_resolution_str == '10K':
        raw_resolution = 10000
    elif raw_resolution_str is None:
        raw_resolution = None
    else:
        raise NotImplementedError
    if (raw_resolution is not None) and (not shuffle):
        cell_table_raw = make_raw_matrix_cell_table(cell_table, raw_resolution_str)
        raw_dir = output_dir / 'raw'
        raw_dir.mkdir(exist_ok=True)
        raw_table_path = raw_dir / 'cell_table.tsv'
        cell_table_raw.to_csv(raw_table_path, sep='\t', header=None)
        merge_raw_cmd = f'hic-internal merge-raw-scool ' \
                        f'--chrom_size_path {chrom_size_path} ' \
                        f'--resolution {raw_resolution} ' \
                        f'--cell_table_path {raw_table_path} ' \
                        f'--output_dir {raw_dir} ' \
                        f'--cpu {cpu_per_job}'
    else:
        merge_raw_cmd = ''

    if log_e:
        log_e_str = '--log_e'
    else:
        log_e_str = ''
    chunk_parameters = dict(
        dist=dist,
        cap=cap,
        pad=pad,
        gap=gap,
        resolution=resolution,
        min_cutoff=min_cutoff,
        chrom_size_path=chrom_size_path,
        keep_cell_matrix=keep_cell_matrix,
        log_e_str=log_e_str,
        shuffle=shuffle
    )

    total_chunk_dirs = []
    group_chunks = {}
    for group, group_df in cell_table.groupby('cell_group'):
        group_chunks[group] = []
        if group_df.shape[0] <= chunk_size:
            this_dir = output_dir / f'{group}_chunk0'
            prepare_dir(this_dir, group_df, **chunk_parameters)
            total_chunk_dirs.append(this_dir)
            group_chunks[group].append(this_dir)
        else:
            group_df['chunk'] = [i // chunk_size for i in range(group_df.shape[0])]
            for chunk, chunk_df in group_df.groupby('chunk'):
                this_dir = output_dir / f'{group}_chunk{chunk}'
                prepare_dir(this_dir, chunk_df, **chunk_parameters)
                total_chunk_dirs.append(this_dir)
                group_chunks[group].append(this_dir)

    with open(output_dir / 'snakemake_cmd_step1.txt', 'w') as f:
        f.write(merge_raw_cmd + '\n')
        for chunk_dir in total_chunk_dirs:
            cmd = f'snakemake -d {chunk_dir} --snakefile {chunk_dir}/Snakefile -j {cpu_per_job} --scheduler greedy --rerun-incomplete'
            f.write(cmd + '\n')

    # prepare the second step that merge cell chunks into groups
    scool_parameters = dict(
        output_dir=f'"{output_dir}"',
        chrom_size_path=f'"{chrom_size_path}"',
        resolution=resolution,
        shuffle=shuffle,
        e_positive_only=e_positive_only,
        use_bkfilter=use_bkfilter
    )
    parameters_str = '\n'.join(f'{k} = {v}'
                               for k, v in scool_parameters.items())
    with open(output_dir / 'Snakefile', 'w') as f:
        f.write(parameters_str + GENERATE_MATRIX_SCOOL_TEMPLATE)
    with open(output_dir / 'snakemake_cmd_step2.txt', 'w') as f:
        cmd = f'snakemake -d {output_dir} --snakefile {output_dir}/Snakefile -j {cpu_per_job} --scheduler greedy --rerun-incomplete'
        f.write(cmd + '\n')
    return


def check_chunk_dir_finish(output_dir):
    output_dir = pathlib.Path(output_dir)
    flag_path = output_dir / 'chunk_finished'
    if flag_path.exists():
        return

    path_not_finished = []
    for chunk_dir in output_dir.glob('*_chunk*'):
        if not (chunk_dir / 'finish').exists():
            path_not_finished.append(str(chunk_dir))
    if len(path_not_finished) > 0:
        path_str = '\n'.join(path_not_finished)
        raise ValueError('These chunk dirs have not finish successfully:\n' + path_str)
    else:
        with open(flag_path, 'w') as _:
            pass
    return


def _unlock_snakemake(output_dir):
    """Unlock all snakemake working directories to allow restart after preemption."""
    output_dir = pathlib.Path(output_dir).absolute()
    # Unlock chunk dirs (step1)
    for chunk_dir in output_dir.glob('*_chunk*'):
        snakefile = chunk_dir / 'Snakefile'
        if snakefile.exists():
            subprocess.run(
                f'snakemake -d {chunk_dir} --snakefile {snakefile} --unlock',
                shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    # Unlock merge dir (step2)
    snakefile = output_dir / 'Snakefile'
    if snakefile.exists():
        subprocess.run(
            f'snakemake -d {output_dir} --snakefile {snakefile} --unlock',
            shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    # Remove stale chunk_finished flag so check_chunk_dir_finish re-validates
    flag = output_dir / 'chunk_finished'
    if flag.exists():
        flag.unlink()


def _run_snakemake(output_dir):
    output_dir = pathlib.Path(output_dir).absolute()
    _unlock_snakemake(output_dir)
    step1 = f'{output_dir}/snakemake_cmd_step1.txt'
    step2 = f'{output_dir}/snakemake_cmd_step2.txt'
    subprocess.run(['sh', step1], check=True)
    subprocess.run(['sh', step2], check=True)


def call_loop(cell_table_path,
              output_dir,
              chrom_size_path,
              shuffle=True,
              chunk_size=200,
              dist=5050000,
              cap=5,
              pad=5,
              gap=2,
              resolution=10000,
              min_cutoff=1e-6,
              keep_cell_matrix=False,
              cpu_per_job=10,
              log_e=True,
              raw_resolution_str=None,
              downsample_shuffle=None,
              black_list_path=None,
              fdr_pad=7,
              fdr_min_dist=5,
              fdr_max_dist=500,
              fdr_thres=0.1,
              dist_thres=20000,
              size_thres=1,
              e_positive_only=True,
              use_bkfilter=True,
              cleanup_totalloop_info=True,
              cleanup=True):
    if shuffle and (black_list_path is None):
        raise ValueError('Please provide black_list_path when shuffle=True')

    pathlib.Path(output_dir).mkdir(exist_ok=True)

    # Whole-pipeline idempotency: if a previous run already wrote Success,
    # there is nothing to do. Without this guard, a requeue would re-enter
    # the FDR step and crash on an already-cleaned-up totalloop_info.hdf.
    if os.path.exists(f'{output_dir}/Success'):
        return

    # deal with cell table path, if the path has two col, add one group cal internally
    _cell_table_path = str(cell_table_path)
    sep = '\t' if _cell_table_path.endswith('tsv') else ','
    cell_table = pd.read_csv(cell_table_path, index_col=0, sep=sep, header=None)
    cell_table.index.name = 'cell_id'
    if cell_table.shape[1] == 1:
        cell_table.columns = ['cell_url']
        cell_table['cell_group'] = 'group'
    elif cell_table.shape[1] == 2:
        cell_table.columns = ['cell_url', 'cell_group']
    else:
        raise ValueError(f'Expect cell_table_path to be '
                         f'two columns (cell_id, cell_url) or '
                         f'three columns (cell_id, cell_url, cell_group), '
                         f'got {cell_table.shape[1]} columns.')
    cell_table_path = f'{output_dir}/cell_table.csv'
    cell_table.to_csv(cell_table_path, header=False, index=True)
    groups = cell_table['cell_group'].unique()

    kwargs = locals()
    shuffle = kwargs.pop('shuffle')

    # prepare snakemake and execute
    real_dir = None
    shuffle_dir = None
    _use_kwargs = {k: v
                   for k, v in kwargs.items()
                   if k in inspect.signature(prepare_loop_snakemake).parameters}
    if shuffle:
        output_dir = _use_kwargs.pop('output_dir')
        real_dir = output_dir
        shuffle_dir = f'{output_dir}/shuffle'
        pathlib.Path(shuffle_dir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(f'{real_dir}/finish'):
            prepare_loop_snakemake(shuffle=False, output_dir=real_dir, **_use_kwargs)
            _run_snakemake(real_dir)
        if not os.path.exists(f'{shuffle_dir}/finish'):
            prepare_loop_snakemake(shuffle=True, output_dir=shuffle_dir, **_use_kwargs)
            _run_snakemake(shuffle_dir)
    else:
        # not shuffle, just use normal loop pipeline
        output_dir = _use_kwargs.pop('output_dir')
        if not os.path.exists(f'{output_dir}/finish'):
            prepare_loop_snakemake(shuffle=False, **_use_kwargs)
            _run_snakemake(output_dir)

    # final call loop if shuffle
    if shuffle:
        for group in groups:
            real_group_prefix = f'{real_dir}/{group}/{group}'
            shuffle_group_prefix = f'{shuffle_dir}/{group}/{group}'

            # Per-group idempotency: FDR + cleanup leaves loop_summit.bedpe
            # in place and removes totalloop_info.hdf. That combination can
            # only occur if the full sequence -- update_fdr_qval,
            # filter_loops, cleanup_totalloop_info -- already ran for this
            # group. Re-running update_fdr_qval here would crash with
            # FileNotFoundError on the deleted hdf.
            # Note: loop_summit.bedpe alone is not enough (it's also written
            # by the initial call_loops() during snakemake step2, before
            # FDR runs); the missing totalloop_info.hdf is what proves the
            # post-FDR cleanup ran.
            if (os.path.exists(f'{real_group_prefix}.loop_summit.bedpe')
                    and not os.path.exists(f'{real_group_prefix}.totalloop_info.hdf')):
                continue

            tot = compute_t(real_group_prefix)
            _ = compute_t(shuffle_group_prefix, tot)

            permute_fdr(chrom_size_path=chrom_size_path,
                        black_list_path=black_list_path,
                        shuffle_group_prefix=shuffle_group_prefix,
                        real_group_prefix=real_group_prefix,
                        res=resolution,
                        pad=fdr_pad,
                        min_dist=fdr_min_dist,
                        max_dist=fdr_max_dist)

            total_loops = update_fdr_qval(chrom_size_path,
                                          real_group_prefix,
                                          shuffle_group_prefix,
                                          res=resolution,
                                          min_dist=fdr_min_dist,
                                          max_dist=fdr_max_dist)

            # redo filter loops because FDR changed
            filter_loops(total_loops,
                         output_prefix=real_group_prefix,
                         fdr_thres=fdr_thres,
                         resolution=resolution,
                         dist_thres=dist_thres,
                         size_thres=size_thres,
                         use_bkfilter=use_bkfilter)

    if cleanup:
        subprocess.run(f'rm -rf {output_dir}/shuffle/*/*.npz', shell=True)
        subprocess.run(f'rm -rf {output_dir}/*/*.npz', shell=True)
    if cleanup_totalloop_info:
        for group in groups:
            real_total = f'{output_dir}/{group}/{group}.totalloop_info.hdf'
            if os.path.exists(real_total):
                os.remove(real_total)
            if shuffle:
                shuffle_total = f'{output_dir}/shuffle/{group}/{group}.totalloop_info.hdf'
                if os.path.exists(shuffle_total):
                    os.remove(shuffle_total)

    with open(f'{output_dir}/Success', 'w') as f:
        f.write('42')
    return


def call_loop_pseudobulk(cell_table_path,
                         output_dir,
                         chrom_size_path,
                         shuffle=True,
                         chunk_size=200,
                         dist=5050000,
                         cap=5,
                         pad=5,
                         gap=2,
                         resolution=10000,
                         min_cutoff=1e-6,
                         keep_cell_matrix=False,
                         cpu_per_job=10,
                         log_e=True,
                         raw_resolution_str=None,
                         cleanup=True):
    """Generate group-level pseudobulk .cool files, without loop calling.

    Runs step1 (per-cell loop matrices) and step2 (chunk->group merge) on the
    real cell-group assignments. Produces
      {output_dir}/{group}/{group}.{E,E2,T,T2,Q,Q2}.cool
    per group. No .bedpe, no FDR.

    If shuffle=True (default), additionally builds a per-diagonal shuffled
    version of every cell's imputed matrix and aggregates those into
      {output_dir}/shuffle/{group}/{group}.{E,E2,T,T2}.cool
    These shuffled group cools are what a downstream permutation-FDR loop
    caller needs. They can only be constructed from single-cell imputed
    cools (the shuffle is per-diagonal per-cell), so building them now,
    together with the real cools, spares you from keeping single-cell
    imputed cools around indefinitely.

    Use call_loop_pseudobulk(shuffle=False) when you'll never do FDR-based
    loop calling on these pseudobulks -- for example, if the downstream
    analysis is a direct A-vs-B pixel comparison.

    Idempotent: if {output_dir}/Success exists, returns immediately.
    """
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    if os.path.exists(f'{output_dir}/Success'):
        return

    # Normalize cell_table (same handling as call_loop).
    _cell_table_path = str(cell_table_path)
    sep = '\t' if _cell_table_path.endswith('tsv') else ','
    cell_table = pd.read_csv(cell_table_path, index_col=0, sep=sep, header=None)
    cell_table.index.name = 'cell_id'
    if cell_table.shape[1] == 1:
        cell_table.columns = ['cell_url']
        cell_table['cell_group'] = 'group'
    elif cell_table.shape[1] == 2:
        cell_table.columns = ['cell_url', 'cell_group']
    else:
        raise ValueError(f'Expect cell_table_path to be '
                         f'two columns (cell_id, cell_url) or '
                         f'three columns (cell_id, cell_url, cell_group), '
                         f'got {cell_table.shape[1]} columns.')
    cell_table_path = f'{output_dir}/cell_table.csv'
    cell_table.to_csv(cell_table_path, header=False, index=True)
    groups = list(cell_table['cell_group'].unique())

    prep_kwargs = dict(cell_table_path=cell_table_path,
                       chrom_size_path=chrom_size_path,
                       chunk_size=chunk_size,
                       dist=dist,
                       cap=cap,
                       pad=pad,
                       gap=gap,
                       resolution=resolution,
                       min_cutoff=min_cutoff,
                       keep_cell_matrix=keep_cell_matrix,
                       cpu_per_job=cpu_per_job,
                       log_e=log_e,
                       raw_resolution_str=raw_resolution_str)

    # --- Real pseudobulk cools ---
    # Prepare with shuffle=False (real data). The generated top-level
    # Snakefile's `summary` rule requires .loop.bedpe (which would trigger
    # the expensive call_loop rule), so we run step1 as usual but launch
    # step2 with the .cool files as explicit targets, bypassing summary.
    if not os.path.exists(f'{output_dir}/finish'):
        prepare_loop_snakemake(output_dir=output_dir, shuffle=False, **prep_kwargs)
        _unlock_snakemake(pathlib.Path(output_dir).absolute())
        subprocess.run(['sh', f'{output_dir}/snakemake_cmd_step1.txt'],
                       check=True)
        matrix_types = ['E', 'E2', 'T', 'T2', 'Q', 'Q2']
        targets = [f'{g}/{g}.{mt}.cool' for g in groups for mt in matrix_types]
        subprocess.run(['snakemake',
                        '-d', str(pathlib.Path(output_dir).absolute()),
                        '--snakefile', f'{output_dir}/Snakefile',
                        '-j', str(cpu_per_job),
                        '--scheduler', 'greedy',
                        '--rerun-incomplete', '--'] + targets,
                       check=True)
        # The `summary` rule's rm -rf never fired because we bypassed it;
        # clean chunk dirs manually and touch the finish sentinel.
        for chunk_dir in pathlib.Path(output_dir).glob('*_chunk*'):
            subprocess.run(f'rm -rf {chunk_dir}', shell=True, check=False)
        pathlib.Path(f'{output_dir}/finish').touch()

    # --- Shuffle pseudobulk cools (optional) ---
    # Prepared with shuffle=True: calculate-loop-matrix is invoked with
    # --shuffle, which per-diagonally permutes each cell's imputed matrix
    # before computing E and T. The step2 Snakefile's `summary` rule then
    # only asks for {E,E2,T,T2}.cool (Q/Q2 are dropped when shuffle=True),
    # so _run_snakemake can run it end-to-end without an explicit-target
    # workaround.
    if shuffle:
        shuffle_dir = f'{output_dir}/shuffle'
        pathlib.Path(shuffle_dir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(f'{shuffle_dir}/finish'):
            prepare_loop_snakemake(output_dir=shuffle_dir, shuffle=True, **prep_kwargs)
            _run_snakemake(shuffle_dir)

    if cleanup:
        subprocess.run(f'rm -rf {output_dir}/shuffle/*/*.npz', shell=True)
        subprocess.run(f'rm -rf {output_dir}/*/*.npz', shell=True)

    with open(f'{output_dir}/Success', 'w') as f:
        f.write('42')
    return


def merge_loop(group,
               output_dir,
               chrom_size_path,
               shuffle=True,
               chunk_size=200,
               dist=5050000,
               cap=5,
               pad=5,
               gap=2,
               resolution=10000,
               min_cutoff=1e-6,
               keep_cell_matrix=False,
               cpu_per_job=10,
               log_e=True,
               raw_resolution_str=None,
               downsample_shuffle=None,
               black_list_path=None,
               fdr_pad=7,
               fdr_min_dist=5,
               fdr_max_dist=500,
               fdr_thres=0.1,
               dist_thres=20000,
               size_thres=1,
               e_positive_only=True,
               use_bkfilter=True,
               cleanup_totalloop_info=True,
               cleanup=True):

    group_list = pd.read_csv(f'{output_dir}/group_list.txt', header=None, index_col=None)[0].values

    if len(group_list)==1:
        tmp = group_list[0].split('/')[-1]
        pathlib.Path(f'{output_dir}/{group}').mkdir(exist_ok=True, parents=True)
        for xx in pathlib.Path(f'{group_list[0]}/{tmp}').iterdir():
            subprocess.run(f'rsync -arv {group_list[0]}/{tmp}/{xx.name} {output_dir}/{group}/{xx.name.replace(tmp, group)}', shell=True)
        pathlib.Path(f'{output_dir}/shuffle/{group}').mkdir(exist_ok=True, parents=True)
        for xx in pathlib.Path(f'{group_list[0]}/shuffle/{tmp}').iterdir():
            subprocess.run(f'rsync -arv {group_list[0]}/shuffle/{tmp}/{xx.name} {output_dir}/shuffle/{group}/{xx.name.replace(tmp, group)}', shell=True)
        with open(f'{output_dir}/Success', 'w') as f:
            f.write('42')
        return
    
    cell_table = pd.concat([pd.read_csv(f'{xx}/cell_table.csv', index_col=0, header=None, 
                                        names=['cell_id', 'cell_url', 'cell_group']) 
                            for xx in group_list], axis=0)
    cell_table['cell_group'] = group
    cell_table.to_csv(f'{output_dir}/cell_table.csv', header=False, index=True)
    kwargs = locals()

    _merge_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(merge_group_to_bigger_group_cools).parameters}
    _merge_kwargs['shuffle'] = False
#     print(_merge_kwargs)
    merge_group_to_bigger_group_cools(**_merge_kwargs)
    
    _loop_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(call_loops).parameters}
#     print(_loop_kwargs)
    call_loops(group_prefix=f'{output_dir}/{group}/{group}', output_prefix=f'{output_dir}/{group}/{group}', **_loop_kwargs)
    
    # prepare snakemake and execute
    if shuffle:
        real_dir = output_dir
        shuffle_dir = f'{output_dir}/shuffle'
        pathlib.Path(shuffle_dir).mkdir(exist_ok=True, parents=True)
        if cell_table.shape[0]>downsample_shuffle:
            _prep_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(prepare_loop_snakemake).parameters}
            _prep_kwargs['output_dir'] = shuffle_dir
            prepare_loop_snakemake(cell_table_path=f'{output_dir}/cell_table.csv', **_prep_kwargs)
            _run_snakemake(shuffle_dir)
        else:
            _merge_kwargs['shuffle'] = True
            _merge_kwargs['output_dir'] = shuffle_dir
            # Shuffle-side cools only exist for E/E2/T/T2 (Q/Q2 are not
            # generated when the chunk Snakefile runs in shuffle mode).
            # Without this override, the default matrix_types tuple would
            # glob for *.Q.cool inside each fine dir's shuffle/ subtree and
            # raise IndexError when the file isn't there.
            _merge_kwargs['matrix_types'] = ('E', 'E2', 'T', 'T2')
            merge_group_to_bigger_group_cools(**_merge_kwargs)
            
        real_group_prefix = f'{real_dir}/{group}/{group}'
        shuffle_group_prefix = f'{shuffle_dir}/{group}/{group}'

        tot = compute_t(real_group_prefix)
        _ = compute_t(shuffle_group_prefix, tot)

        permute_fdr(chrom_size_path=chrom_size_path,
                    black_list_path=black_list_path,
                    shuffle_group_prefix=shuffle_group_prefix,
                    real_group_prefix=real_group_prefix,
                    res=resolution,
                    pad=fdr_pad,
                    min_dist=fdr_min_dist,
                    max_dist=fdr_max_dist)

        total_loops = update_fdr_qval(chrom_size_path,
                                      real_group_prefix,
                                      shuffle_group_prefix,
                                      res=resolution,
                                      min_dist=fdr_min_dist,
                                      max_dist=fdr_max_dist)

        # redo filter loops because FDR changed
        filter_loops(total_loops,
                     output_prefix=real_group_prefix,
                     fdr_thres=fdr_thres,
                     resolution=resolution,
                     dist_thres=dist_thres,
                     size_thres=size_thres,
                     use_bkfilter=use_bkfilter)

    if cleanup:
        subprocess.run(f'rm -rf {output_dir}/*/*.npz', shell=True)
        subprocess.run(f'rm -rf {output_dir}/shuffle/*/*.npz', shell=True)
    if cleanup_totalloop_info:
        real_total = f'{output_dir}/{group}/{group}.totalloop_info.hdf'
        if os.path.exists(real_total):
            os.remove(real_total)
        if shuffle:
            shuffle_total = f'{output_dir}/shuffle/{group}/{group}.totalloop_info.hdf'
            if os.path.exists(shuffle_total):
                os.remove(shuffle_total)

    with open(f'{output_dir}/Success', 'w') as f:
        f.write('42')
