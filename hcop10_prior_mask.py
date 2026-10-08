# -*- coding: utf-8 -*-
# HCO+ (1-0) 的 CO 2-1 prior 3D masking —— co21_prior_3d_masking.ipynb 流程：
#   weak_line=False -> 亮线（同 HCN 1-0）：纯 CO prior
#   weak_line=True  -> 弱线（同 CS 2-1/HCO+ 4-3）：CO 先验(空间 bound) ∩ 目标线
#                      “峰锚定连通”自掩。hcop10 网格/beam 与 cs21 相近
#                      (360x360, 0.1"/pix, beam 9.1 pix)，envelope/腐蚀参数沿用 cs21。
# 两版输出用 tag 区分（co_prior / selfmask），均保存 mask 后的 cube（外置 NaN）。
import os
import numpy as np
import astropy.units as u
from scipy import ndimage
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel
from spectral_cube import SpectralCube

refil  = 'ngc3351_12m+7m+tp_co21.fits'
spefi_ = 'M95_C5+C2_hcop10_pbcorr_round_k.fits'
refpng = 'M95_C5+C2_hcn10_prior_moment0.png'

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
# 主流程（notebook cell 12；tag 区分两版输出；新增 masked cube 写盘）
# ---------------------------------------------------------------------------
def main(weak_line, tag):
    workdir = '.'
    outmomt = f'M95_C5+C2_hcop10_{tag}_mom0.fits'
    outmask = ('co_mask_on_hcop10_grid.fits' if tag == 'co_prior'
               else f'co_{tag}_on_hcop10_grid.fits')
    outcube = ('M95_C5+C2_hcop10_masked_cube.fits' if tag == 'co_prior'
               else f'M95_C5+C2_hcop10_{tag}_masked_cube.fits')
    print('=' * 60)
    print(f'weak_line={weak_line}  tag={tag}')

    co    = SpectralCube.read(f'{workdir}/{refil}').with_spectral_unit(u.km / u.s)
    spefi = SpectralCube.read(f'{workdir}/{spefi_}').with_spectral_unit(u.km / u.s)

    vel = spefi.spectral_axis.value
    dv_med = abs(np.median(np.diff(vel)))
    print(f'目标 cube: {spefi_}')
    print(f'  通道数 {len(vel)}, dv = {dv_med:.2f} km/s, '
          f'速度覆盖 {vel.min():.1f} ~ {vel.max():.1f} km/s')
    print(f'  BMAJ = {spefi.header["BMAJ"]*3600:.2f}", '
          f'beam = {spefi.header["BMAJ"]/abs(spefi.header["CDELT1"]):.1f} pix')

    # --- 1. line-free 通道：按速度窗取补集（与 notebook 的 cs21 定义一致）---
    is_line = (vel >= V_LINE_MIN) & (vel <= V_LINE_MAX)
    tgt_line_free = np.where(~is_line)[0]
    co_line_free = np.r_[0:84, 199:275]
    print(f'  line-free 通道: 共 {len(tgt_line_free)} ch')

    # --- 2. CO 先验 mask（亮线 6σ/4σ）并投影到 hcop10 网格 ---
    co_s = smooth_3d(co, spatial_fwhm_pix=8, spectral_fwhm_chan=4)
    noise_co = estimate_noise_per_channel(co_s, co_line_free)
    mask_co = hierarchical_mask(co_s, noise_co,
                                core_sigma=6.0, low_sigma=4.0,
                                min_channels=2, min_pixels=beam_area_pixels(co))
    mask_prior = transfer_mask(mask_co, co_s, spefi)

    # 发射线速度窗，掩掉线外通道防止先验在 line-free 区泄漏
    mask_prior &= is_line[:, None, None]

    if weak_line:
        # --- 弱线（同 cs21）：CO 先验 ∩ 目标线“峰锚定连通”自掩 ---
        # hcop10 beam = 9.1 pix，平滑取 ~1 beam = 9 pix + 2 通道（notebook:
        # cs21 beam 8.5 用 8 pix；hcop43 beam 6.98 用 7 pix）
        tgt_s = smooth_3d(spefi, spatial_fwhm_pix=9, spectral_fwhm_chan=2)
        noise_tgt = estimate_noise_per_channel(tgt_s, tgt_line_free)
        snr_t = tgt_s.filled_data[:].value / noise_tgt.value[:, None, None]

        # 1) core(峰,3σ) / low(低阈支撑) 两个 3D 掩膜，限制在 CO 先验内
        core_sigma, low_sigma, low_sigma_in = 3.0, 1.0, 0.5
        core3d = (snr_t >= core_sigma) & mask_prior

        # 1b) 环 envelope（CO 先验 2D 支撑闭合+填洞, r<95 裁切）内 low 降阈，
        #     沿用 cs21 的 0.5σ（hcop10 环内弥漫同样暗弱）
        co2d_any = mask_prior.any(axis=0)
        _closed = ndimage.binary_closing(co2d_any,
                                          structure=np.ones((25, 25)))
        inner_env = ndimage.binary_fill_holes(_closed)
        _nyi, _nxi = inner_env.shape
        _yyi, _xxi = np.mgrid[0:_nyi, 0:_nxi]
        inner_env &= np.hypot(_yyi - (_nyi - 1) / 2.0,
                               _xxi - (_nxi - 1) / 2.0) < 95.0
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

        # 5) 最外缘向里腐蚀 3 pix 剥掉锯齿状弱边（cs21 同参数），
        #    腐蚀新封出的内洞填回，原本空的内洞保持不掩
        _fp_edge = mask_final.any(axis=0)
        _fp_eroded = ndimage.binary_erosion(
            _fp_edge, structure=st2, iterations=3)
        _new_holes = ndimage.binary_fill_holes(_fp_eroded) & ~_fp_eroded
        _fp_eroded = _fp_eroded | (_new_holes & _fp_edge)
        print(f'外围腐蚀: {_fp_edge.sum():,} -> {_fp_eroded.sum():,} 像素 '
              f'(剥除 {_fp_edge.sum() - _fp_eroded.sum():,}, '
              f'填回内洞 {(_new_holes & _fp_edge).sum():,})')
        mask_final = mask_final & _fp_eroded[None, :, :]
    else:
        # 亮线（HCN 1-0 方案）：纯 CO prior
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

    # --- 4. 写盘 ---
    hdr2d = spefi.wcs.sub(['longitude', 'latitude']).to_header()
    hdr2d['BUNIT'] = 'K km/s'
    hdr3d = spefi.wcs.to_header()
    for _k in ('BMAJ', 'BMIN', 'BPA'):      # beam 信息写入所有产品头
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
    # mask 内总流量（两版对比用）
    print(f'掩内 mom0 总和 = {np.nansum(mom0):.0f} K km/s pix')

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

    # --- 6. 对比图 ---
    import matplotlib.image as mpimg
    refimg = mpimg.imread(refpng)
    _sm = fits.getdata(outmomt.replace('.fits', '_sm1beam.fits'))
    _vmax1 = float(np.nanpercentile(_sm, 99.5))

    if weak_line:
        # 弱线版：[纯 prior mom0(若存在)] | 自掩 native | 自掩 1-beam | 参考图
        _prior_f = 'M95_C5+C2_hcop10_co_prior_mom0.fits'
        _has_prior = os.path.exists(_prior_f)
        _np = 4 if _has_prior else 3
        _vmax0 = float(np.nanpercentile(mom0, 99.5))
        fig, a = plt.subplots(1, _np, figsize=(5.5 * _np, 5.5))
        _i = 0
        if _has_prior:
            prior_mom0 = fits.getdata(_prior_f)
            im = a[0].imshow(prior_mom0, origin='lower', cmap='magma', vmin=0,
                             vmax=float(np.nanpercentile(prior_mom0, 99.5)))
            a[0].set_title('pure CO prior mom0 [K km/s]')
            plt.colorbar(im, ax=a[0], shrink=.8)
            _i = 1
        im = a[_i].imshow(mom0, origin='lower', cmap='magma', vmin=0, vmax=_vmax0)
        a[_i].set_title('weak-line selfmask mom0\nstretch 0-%.0f' % _vmax0)
        plt.colorbar(im, ax=a[_i], shrink=.8)
        im = a[_i + 1].imshow(_sm, origin='lower', cmap='magma', vmin=0,
                              vmax=_vmax1)
        a[_i + 1].set_title('selfmask 1-beam smoothed\nstretch 0-%.0f' % _vmax1)
        plt.colorbar(im, ax=a[_i + 1], shrink=.8)
        a[_i + 2].imshow(refimg)
        a[_i + 2].set_title(refpng + '\n(ref: bottom-middle hcop10 panel)')
        a[_i + 2].axis('off')
    else:
        mom0_chk = fits.getdata(outmomt)
        _vmax0 = float(np.nanpercentile(mom0_chk, 99.5))
        fig, a = plt.subplots(1, 3, figsize=(18, 6))
        im0 = a[0].imshow(mom0_chk, origin='lower', cmap='magma', vmin=0,
                          vmax=_vmax0)
        a[0].set_title(outmomt + '\nnative [K km/s], stretch 0-%.0f' % _vmax0)
        plt.colorbar(im0, ax=a[0], shrink=.8)
        im1 = a[1].imshow(_sm, origin='lower', cmap='magma', vmin=0, vmax=_vmax1)
        a[1].set_title('1-beam smoothed, full mask\nstretch 0-%.0f' % _vmax1)
        plt.colorbar(im1, ax=a[1], shrink=.8)
        a[2].imshow(refimg)
        a[2].set_title(refpng + '\n(ref: bottom-middle hcop10 panel)')
        a[2].axis('off')
    plt.tight_layout()
    _outcmp = outmomt.replace('.fits', '_compare.png')
    plt.savefig(_outcmp, dpi=130)
    plt.close()
    print('saved', _outcmp)


if __name__ == '__main__':
    # selfmask 为最终采用方案（hcop10 为弱线）；co_prior 版仅历史对比用，
    # 其产品已删除。如需复现对比，手动调用 main(weak_line=False, tag='co_prior')
    main(weak_line=True, tag='selfmask')
