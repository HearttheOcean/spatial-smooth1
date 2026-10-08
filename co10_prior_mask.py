# -*- coding: utf-8 -*-
# CO 1-0 的 CO 2-1 prior 3D masking —— co21_prior_3d_masking.ipynb 流程：
#   weak_line=False -> 亮线（同 HCN 1-0 / CO 3-2）：纯 CO prior
#   weak_line=True  -> 弱线自掩：CO 先验(空间 bound) ∩ 目标线“峰锚定连通”自掩。
#                      CO 1-0 亮环很亮，自掩主要作用是剥掉先验覆盖但无 CO 1-0
#                      信号的外围区域（那里 mom0≈0 只贡献噪声碎斑）。
# 参数按分辨率缩放（hcop10: 0.1"/pix, beam 9.1 pix；co10: 0.6"/pix, beam 3.3 pix）：
#   平滑 ~1 beam = 3 pix + 2 ch；envelope r<9.5" = 16 pix；closing 2.5" = 4 pix；
#   外缘腐蚀 beam/3 ≈ 1 pix。
# 输出（tag 区分）：mask、masked cube（外置 NaN）、mom0/rms/snr/sm1beam（含 beam 头）。
import os
import numpy as np
import astropy.units as u
from scipy import ndimage
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel
from spectral_cube import SpectralCube

refil  = 'ngc3351_12m+7m+tp_co21.fits'
spefi_ = 'ngc3351_12m+7m+tp_co10_pbcorr_trimmed_k.fits'

V_LINE_MIN, V_LINE_MAX = 625.0, 918.0     # 发射线速度窗 [km/s]，与 notebook 统一


# ---------------------------------------------------------------------------
# 以下为 notebook cells 1-5 的辅助函数（逐字复制，未改动）
# ---------------------------------------------------------------------------
def estimate_noise_per_channel(cube, line_free_channels):
    """用 line-free 通道的 MAD 估计逐通道噪声 (与 cube 同单位)。"""
    data = cube.filled_data[:].value          # NaN 已填充
    noise = np.full(cube.shape[0], np.nan)
    for i in line_free_channels:
        plane = data[i][np.isfinite(data[i])]
        noise[i] = 1.4826 * np.median(np.abs(plane - np.median(plane)))
    good = np.isfinite(noise)
    noise = np.interp(np.arange(cube.shape[0]), np.where(good)[0], noise[good])
    return noise * cube.unit


def smooth_3d(cube, spatial_fwhm_pix, spectral_fwhm_chan):
    """空间 + 谱向高斯平滑。平滑核宽度应 ~beam / ~线宽，而不是固定 5 pixel。"""
    k_sp = Gaussian2DKernel(spatial_fwhm_pix / 2.3548)
    k_v  = Gaussian1DKernel(spectral_fwhm_chan / 2.3548)
    return cube.spatial_smooth(k_sp).spectral_smooth(k_v)


def hierarchical_mask(cube, noise, core_sigma=4.0, low_sigma=2.0,
                      min_channels=2, min_pixels=None, bound=None):
    """在 *平滑后* 的 cube 上生成 3D mask（双阈值区域生长）。"""
    data = cube.filled_data[:].value
    noise_arr = noise.to(cube.unit).value[:, None, None]

    snr = data / noise_arr
    core = snr >= core_sigma
    low  = snr >= low_sigma
    if bound is not None:
        core &= bound
        low  &= bound

    struct = ndimage.generate_binary_structure(3, 3)
    grown = ndimage.binary_propagation(core, mask=low, structure=struct)

    lab, n = ndimage.label(grown, structure=struct)
    if min_pixels is None:
        min_pixels = 1
    counts = np.bincount(lab.ravel())
    spec_span = np.zeros(n + 1, dtype=int)
    for s in range(cube.shape[0]):
        ids = np.unique(lab[s][lab[s] > 0])
        spec_span[ids] += 1
    ok = (counts >= min_pixels) & (spec_span >= min_channels)
    ok[0] = False
    keep = ok[lab]
    return keep


def beam_area_pixels(cube):
    """beam 面积（像素数），作为 min_pixels 的物理下限。"""
    from radio_beam import Beam
    beam = Beam.from_fits_header(cube.header)
    pix = abs(cube.header['CDELT1']) * u.deg
    return int(np.ceil((beam.sr / (pix**2).to(u.sr)).value))


def transfer_mask(mask_bool, co_cube, tgt_cube):
    """把 CO 网格上的布尔 mask 重采样到目标 cube 网格（谱向速度插值 + 空间 WCS 重投影）。"""
    from astropy.coordinates import SkyCoord
    from scipy.ndimage import map_coordinates

    v_co = co_cube.spectral_axis.to(u.km / u.s).value
    v_t  = tgt_cube.spectral_axis.to(u.km / u.s).value
    m = mask_bool.astype(np.float32)

    order = np.argsort(v_co)
    v_sorted, m_sorted = v_co[order], m[order]
    idx = np.interp(v_t, v_sorted, np.arange(len(v_sorted)),
                    left=-1, right=-1)
    spec_resampled = np.zeros((len(v_t),) + m.shape[1:], dtype=np.float32)
    for j, fi in enumerate(idx):
        if fi < 0:
            continue
        i0 = int(np.floor(fi))
        i1 = min(i0 + 1, len(v_sorted) - 1)
        w = fi - i0
        spec_resampled[j] = (1 - w) * m_sorted[i0] + w * m_sorted[i1]

    ny, nx = tgt_cube.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    sc = tgt_cube.wcs.celestial.pixel_to_world(xx.ravel(), yy.ravel())
    frame = co_cube.wcs.celestial.wcs.radesys
    if isinstance(frame, bytes):
        frame = frame.decode()
    frame = {'ICRS': 'icrs', 'FK5': 'fk5', 'FK4': 'fk4'}.get(
        str(frame).strip(), 'icrs')
    co_sc = sc.transform_to(frame)
    cx, cy = co_cube.wcs.celestial.world_to_pixel(co_sc)
    coords = np.vstack([cy.reshape(1, -1), cx.reshape(1, -1)])

    out = np.zeros((len(v_t), ny, nx), dtype=np.float32)
    for j in range(len(v_t)):
        p = map_coordinates(spec_resampled[j], coords, order=1,
                            mode='constant', cval=0.0)
        out[j] = p.reshape(ny, nx)
    return out >= 0.5


# ---------------------------------------------------------------------------
# 主流程（notebook cell 13；tag 区分两版输出；masked cube 写盘 + beam 头）
# ---------------------------------------------------------------------------
def main(weak_line, tag):
    workdir = '.'
    outmomt = f'ngc3351_12m+7m+tp_co10_{tag}_mom0.fits'
    outmask = ('co_mask_on_co10_grid.fits' if tag == 'co_prior'
               else f'co_{tag}_on_co10_grid.fits')
    outcube = ('ngc3351_12m+7m+tp_co10_masked_cube.fits' if tag == 'co_prior'
               else f'ngc3351_12m+7m+tp_co10_{tag}_masked_cube.fits')
    print('=' * 60)
    print(f'weak_line={weak_line}  tag={tag}')

    co    = SpectralCube.read(f'{workdir}/{refil}').with_spectral_unit(u.km / u.s)
    spefi = SpectralCube.read(f'{workdir}/{spefi_}').with_spectral_unit(u.km / u.s)

    vel = spefi.spectral_axis.value
    print(f'目标 cube: {spefi_}')
    print(f'  通道数 {len(vel)}, dv = {abs(np.median(np.diff(vel))):.2f} km/s, '
          f'速度覆盖 {vel.min():.1f} ~ {vel.max():.1f} km/s')
    print(f'  BMAJ = {spefi.header["BMAJ"]*3600:.2f}", '
          f'beam = {spefi.header["BMAJ"]/abs(spefi.header["CDELT1"]):.1f} pix')

    # --- 1. line-free 通道：按速度窗取补集（notebook cell 11 定义）---
    is_line = (vel >= V_LINE_MIN) & (vel <= V_LINE_MAX)
    tgt_line_free = np.where(~is_line)[0]
    co_line_free = np.r_[0:84, 199:275]
    print(f'  line-free 通道: 共 {len(tgt_line_free)} ch')

    # --- 2. CO 先验 mask（亮线 6σ/4σ）并投影到 co10 网格 ---
    co_s = smooth_3d(co, spatial_fwhm_pix=8, spectral_fwhm_chan=4)
    noise_co = estimate_noise_per_channel(co_s, co_line_free)
    mask_co = hierarchical_mask(co_s, noise_co,
                                core_sigma=6.0, low_sigma=4.0,
                                min_channels=2, min_pixels=beam_area_pixels(co))
    mask_prior = transfer_mask(mask_co, co_s, spefi)

    # 发射线速度窗，掩掉线外通道防止先验在 line-free 区泄漏
    mask_prior &= is_line[:, None, None]

    if weak_line:
        # --- 弱线自掩（同 cs21/hcop10 逻辑，参数按 0.6"/pix、beam 3.3 pix 缩放）---
        # 平滑 ~1 beam = 3 pix + 2 通道（notebook: ~1 beam + 2 ch）
        tgt_s = smooth_3d(spefi, spatial_fwhm_pix=3, spectral_fwhm_chan=2)
        noise_tgt = estimate_noise_per_channel(tgt_s, tgt_line_free)
        snr_t = tgt_s.filled_data[:].value / noise_tgt.value[:, None, None]

        # 1) core(峰,3σ) / low(低阈支撑) 两个 3D 掩膜，限制在 CO 先验内
        core_sigma, low_sigma, low_sigma_in = 3.0, 1.0, 0.5
        core3d = (snr_t >= core_sigma) & mask_prior

        # 1b) 环 envelope 内 low 降阈 0.5σ。尺度换算（hcop10 -> co10）：
        #     closing 25 pix(2.5") -> 4 pix；r<95 pix(9.5") -> 16 pix
        co2d_any = mask_prior.any(axis=0)
        _closed = ndimage.binary_closing(co2d_any,
                                          structure=np.ones((4, 4)))
        inner_env = ndimage.binary_fill_holes(_closed)
        _nyi, _nxi = inner_env.shape
        _yyi, _xxi = np.mgrid[0:_nyi, 0:_nxi]
        inner_env &= np.hypot(_yyi - (_nyi - 1) / 2.0,
                               _xxi - (_nxi - 1) / 2.0) < 16.0
        low3d = (((snr_t >= low_sigma) |
                  ((snr_t >= low_sigma_in) & inner_env[None, :, :]))
                 & mask_prior)

        # 2) 峰锚点：core 在 >=2 个通道出现的空间像素
        seed2d = core3d.sum(axis=0) >= 2

        # 3) low 支撑 2D 连通，只保留含峰连通块
        supp2d = low3d.any(axis=0)
        st2 = ndimage.generate_binary_structure(2, 2)
        lab, nlab = ndimage.label(supp2d, structure=st2)
        anchored_ids = np.unique(lab[seed2d & (lab > 0)])
        footprint = np.isin(lab, anchored_ids) & (lab > 0)

        # 4) 3D 逐通道取 low，空间只留峰连通域；要求 >=2 通道
        mask_final = low3d & footprint[None, :, :]
        nchan = mask_final.sum(axis=0)
        mask_final = mask_final & (nchan >= 2)[None, :, :]

        # 5) 外缘腐蚀剥锯齿弱边：hcop10 用 3 pix = beam/3；co10 beam 3.3 pix
        #    -> 1 pix。腐蚀新封出的内洞填回，原本空的内洞保持不掩
        _fp_edge = mask_final.any(axis=0)
        _fp_eroded = ndimage.binary_erosion(
            _fp_edge, structure=st2, iterations=1)
        _new_holes = ndimage.binary_fill_holes(_fp_eroded) & ~_fp_eroded
        _fp_eroded = _fp_eroded | (_new_holes & _fp_edge)
        print(f'外围腐蚀: {_fp_edge.sum():,} -> {_fp_eroded.sum():,} 像素 '
              f'(剥除 {_fp_edge.sum() - _fp_eroded.sum():,}, '
              f'填回内洞 {(_new_holes & _fp_edge).sum():,})')
        mask_final = mask_final & _fp_eroded[None, :, :]
    else:
        # 亮线（HCN 1-0 / CO 3-2 方案）：纯 CO prior
        mask_final = mask_prior

    # --- 3. Moment 0：native cube 显式积分 ---
    data = spefi.filled_data[:].value.copy()
    _goodch = np.isfinite(data).any(axis=(1, 2))
    pb_ok = np.isfinite(data[_goodch]).all(axis=0)
    data = np.where(np.isfinite(data), data, 0.0)
    dv = np.abs(np.gradient(vel))
    mom0 = (np.where(mask_final, data, 0.0) * dv[:, None, None]).sum(axis=0)
    fp2d = mask_final.any(axis=0)
    mom0[~fp2d] = np.nan
    mom0[~pb_ok] = np.nan

    # 原始（未掩）mom0：co10 含 TP，raw/masked 流量应一致，作交叉检验
    mom0_raw = (data * dv[:, None, None]).sum(axis=0)
    mom0_raw[~pb_ok] = np.nan

    # --- 3b. 1-beam 显示产品（footprint 权重归一平滑）---
    from astropy.convolution import convolve
    beam_pix_t = spefi.header['BMAJ'] / np.abs(spefi.header['CDELT1'])
    kern1b = Gaussian2DKernel(beam_pix_t / 2.355)
    _num = convolve(np.where(np.isfinite(mom0), mom0, 0.0), kern1b,
                    boundary='fill', fill_value=0.0, nan_treatment='fill')
    _wgt = convolve(fp2d.astype(float), kern1b,
                    boundary='fill', fill_value=0.0)
    mom0_sm = np.where(_wgt > 0.3, _num / np.where(_wgt > 0.3, _wgt, 1.0),
                       np.nan)
    mom0_sm[~fp2d] = np.nan
    mom0_sm[~pb_ok] = np.nan

    # mom0 逐像素噪声
    rms_native = estimate_noise_per_channel(spefi, tgt_line_free).value
    var = np.zeros_like(mom0)
    for j in range(mask_final.shape[0]):
        var += mask_final[j] * (dv[j] * rms_native[j])**2
    mom0_rms = np.sqrt(var)
    mom0_snr = np.divide(mom0, mom0_rms, out=np.zeros_like(mom0),
                         where=mom0_rms > 0)
    mom0_rms[~fp2d] = np.nan
    mom0_snr[~fp2d] = np.nan
    mom0_rms[~pb_ok] = np.nan
    mom0_snr[~pb_ok] = np.nan

    # --- 4. 写盘（beam 信息写入所有产品头）---
    hdr2d = spefi.wcs.sub(['longitude', 'latitude']).to_header()
    hdr2d['BUNIT'] = 'K km/s'
    hdr3d = spefi.wcs.to_header()
    for _k in ('BMAJ', 'BMIN', 'BPA'):
        hdr2d[_k] = spefi.header[_k]
        hdr3d[_k] = spefi.header[_k]
    fits.writeto(f'{workdir}/{outmomt}', mom0.astype(np.float32),
                 hdr2d, overwrite=True)
    fits.writeto(f'{workdir}/{outmask}', mask_final.astype(np.int16),
                 hdr3d, overwrite=True)
    fits.writeto(f'{workdir}/{outmomt.replace(".fits", "_rms.fits")}',
                 mom0_rms.astype(np.float32), hdr2d, overwrite=True)
    fits.writeto(f'{workdir}/{outmomt.replace(".fits", "_snr.fits")}',
                 mom0_snr.astype(np.float32), hdr2d, overwrite=True)
    fits.writeto(f'{workdir}/{outmomt.replace(".fits", "_sm1beam.fits")}',
                 mom0_sm.astype(np.float32), hdr2d, overwrite=True)

    # mask 后的 cube：mask 外置 NaN，保留原始 3D 头
    data_raw = spefi.filled_data[:].value
    masked_cube = np.where(mask_final, data_raw, np.nan)
    fits.writeto(f'{workdir}/{outcube}', masked_cube.astype(np.float32),
                 spefi.header, overwrite=True)
    print(f'masked cube 已写出: {outcube}')

    # --- 5. 质检 ---
    sp = mask_final.any(axis=0)
    print(f'CO  mask 体素: {mask_co.sum():,}')
    print(f'投影到目标网格: {mask_prior.sum():,} ({mask_prior.mean():.1%} of cube)')
    if weak_line:
        print(f'低阈连通块总数: {nlab:,}, 含峰锚定块: {len(anchored_ids):,}')
        print(f'环 envelope 像素: {inner_env.sum():,}（其内 low={low_sigma_in}sigma）')
    print(f'最终 mask: {mask_final.sum():,} 体素, 空间覆盖 {sp.mean():.1%}')
    print(f'mom0 peak = {np.nanmax(mom0):.2f} K km/s'
          f'   (1-beam 平滑显示产品 peak = {np.nanmax(mom0_sm):.2f})')
    sel = sp & pb_ok
    print(f'掩内 mom0 |S/N| 中位 = {np.nanmedian(np.abs(mom0_snr[sel])):.2f}')
    print(f'|S/N|>3 像素占 FOV: {(sel & (np.abs(mom0_snr) > 3)).mean():.2%}')
    # 含 TP：掩内总流量与同区原始流量应一致（mask 只去噪不改流量）
    print(f'掩内 mom0 总和 = {np.nansum(mom0):.0f} K km/s pix')
    print(f'同区原始总和 = {np.nansum(mom0_raw[fp2d & pb_ok]):.0f} K km/s pix '
          f'(masked/raw = {np.nansum(mom0)/np.nansum(mom0_raw[fp2d & pb_ok]):.3f})')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(1, 2, figsize=(11, 5))
    im0 = a[0].imshow(mom0, origin='lower', cmap='magma')
    a[0].set_title(f'{outmomt}\nmoment 0 [K km/s]')
    plt.colorbar(im0, ax=a[0], shrink=.8)
    im1 = a[1].imshow(mom0_snr, origin='lower', cmap='viridis',
                      vmin=-2, vmax=8)
    a[1].set_title('moment 0 S/N')
    plt.colorbar(im1, ax=a[1], shrink=.8)
    plt.tight_layout()
    plt.savefig(f'{workdir}/{outmomt.replace(".fits", "_qc.png")}', dpi=130)
    plt.close()

    # --- 6. 对比图（末栏固定为 CO 2-1 mom0 形态参照）---
    co_vel = co.spectral_axis.value
    co_dv = np.abs(np.gradient(co_vel))
    co_data = co.filled_data[:].value
    co_data = np.where(np.isfinite(co_data), co_data, 0.0)
    co_mom0 = (np.where(mask_co, co_data, 0.0) * co_dv[:, None, None]).sum(axis=0)
    co_mom0[~mask_co.any(axis=0)] = np.nan

    _vmax0 = float(np.nanpercentile(mom0, 99.5))
    _vmax1 = float(np.nanpercentile(mom0_sm, 99.5))
    fig, a = plt.subplots(1, 4, figsize=(23, 5.8))
    if weak_line:
        # 左栏：已存盘的纯 prior mom0 作对照
        _prior_f = 'ngc3351_12m+7m+tp_co10_co_prior_mom0.fits'
        prior_mom0 = fits.getdata(_prior_f)
        im = a[0].imshow(prior_mom0, origin='lower', cmap='magma', vmin=0,
                         vmax=float(np.nanpercentile(prior_mom0, 99.5)))
        a[0].set_title('pure CO prior mom0')
        plt.colorbar(im, ax=a[0], shrink=.8)
        _t1, _t2 = 'selfmask native', 'selfmask 1-beam smoothed'
    else:
        _vmaxr = float(np.nanpercentile(mom0_raw, 99.5))
        im = a[0].imshow(mom0_raw, origin='lower', cmap='magma', vmin=0,
                         vmax=_vmaxr)
        a[0].set_title('raw (no mask) 625-918 km/s\nstretch 0-%.0f' % _vmaxr)
        plt.colorbar(im, ax=a[0], shrink=.8)
        _t1, _t2 = 'CO prior masked native', 'masked 1-beam smoothed'
    im = a[1].imshow(mom0, origin='lower', cmap='magma', vmin=0, vmax=_vmax0)
    a[1].set_title(_t1 + '\nstretch 0-%.0f' % _vmax0)
    plt.colorbar(im, ax=a[1], shrink=.8)
    im = a[2].imshow(mom0_sm, origin='lower', cmap='magma', vmin=0, vmax=_vmax1)
    a[2].set_title(_t2 + '\nstretch 0-%.0f' % _vmax1)
    plt.colorbar(im, ax=a[2], shrink=.8)
    im = a[3].imshow(co_mom0, origin='lower', cmap='magma', vmin=0,
                     vmax=float(np.nanpercentile(co_mom0, 99.5)))
    a[3].set_title('CO 2-1 mom0 (morphology ref)')
    plt.colorbar(im, ax=a[3], shrink=.8)
    plt.tight_layout()
    _outcmp = outmomt.replace('.fits', '_compare.png')
    plt.savefig(_outcmp, dpi=130)
    plt.close()
    print('saved', _outcmp)


if __name__ == '__main__':
    # CO 1-0 自掩版：剥掉先验内无 CO 1-0 信号的外围噪声区；
    # 纯 prior 版产品已存在（tag='co_prior'），如需重跑手动调用 main(False, 'co_prior')
    main(weak_line=True, tag='selfmask')
