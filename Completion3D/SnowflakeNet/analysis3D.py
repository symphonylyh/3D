
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from scipy.optimize import curve_fit
import pandas as pd 
import openpyxl

class Analysis3D:
    def __init__(self, input_spreadsheet, gt_path, output_path):
        self.input_spreadsheet = input_spreadsheet
        self.filestem = os.path.splitext(os.path.basename(input_spreadsheet))[0]
        self.gt_path = gt_path
        self.output_path = output_path

    def parse_spreadsheet(self):
        # prediction
        pred_df = pd.read_excel(self.input_spreadsheet, sheet_name='Completion')
        pred = pred_df.iloc[:9,1:].dropna().to_numpy()

        # gt
        rr3_part1 = pd.read_excel(os.path.join(self.gt_path, 'RR3.xlsx'), sheet_name='Metashape Stats')
        rr3_part2 = pd.read_excel(os.path.join(self.gt_path, 'RR3.xlsx'), sheet_name='3D Stats')

        rr4_part1 = pd.read_excel(os.path.join(self.gt_path, 'RR4.xlsx'), sheet_name='Metashape Stats')
        rr4_part2 = pd.read_excel(os.path.join(self.gt_path, 'RR4.xlsx'), sheet_name='3D Stats')

        rr3_part1 = rr3_part1.iloc[:, 1:3].dropna().to_numpy() # volume, area
        rr3_part2 = rr3_part2.iloc[:, :].dropna().to_numpy() # ESD, abc, FER, Sphericity
        gt_rr3 = np.concatenate((rr3_part2[:,:4], rr3_part1, rr3_part2[:,4:6]), axis=1)

        rr4_part1 = rr4_part1.iloc[:, 1:3].dropna().to_numpy() # volume, area
        rr4_part2 = rr4_part2.iloc[:, :].dropna().to_numpy() # ESD, abc, FER, Sphericity
        gt_rr4 = np.concatenate((rr4_part2[:,:4], rr4_part1, rr4_part2[:,4:6]), axis=1)

        gt_map = {3: gt_rr3, 4: gt_rr4}

        gt = np.ones_like(pred)
        # -1 means over segmentation, 0 means no visible corresponce from images, ID is 4XX for RR4, 3XX for RR3
        gt_ids = pred_df.iloc[9, 1:].dropna().to_numpy().astype(np.int32)
        for ins_id, gt_id in enumerate(gt_ids):
            if gt_id > 0:
                cat, id = gt_id // 100, gt_id % 100
                gt[:8, ins_id] = gt_map[cat][id-1]

        # only plot those rocks with valid GT correspondence & with success mesh reconstruction (failed reconstruction will save ESD, volume, area as 0)
        ins_mask = np.logical_and(gt_ids > 0, np.logical_and(pred[0] > 1e-4, pred[8] < 0.95)) # shape percentage > 0.95 usually is a planar
        num_oversegmentation = np.count_nonzero(gt_ids == -1)
        num_undersegmentation = np.count_nonzero(gt_ids == -2)
        num_gt_missing = np.count_nonzero(gt_ids == 0)

        pred_valid = pred[:, ins_mask]
        gt_valid = gt[:, ins_mask]
        print(f'{pred_valid.shape[1]} out of {pred.shape[1]} instances have valid ground-truth match. {num_oversegmentation} oversegmented instances, {num_undersegmentation} undersegmented instances, {num_gt_missing} ground-truth not visible.')

        # write to spreasheet
        save_spreadsheet_name = self.filestem + '_benchmark.xlsx'
        spreadsheet = os.path.join(self.output_path, save_spreadsheet_name)
        with pd.ExcelWriter(spreadsheet, mode='w') as writer:
            pd.DataFrame(pred_valid).to_excel(writer, sheet_name='Completion', float_format='%.3f')
            pd.DataFrame(gt_valid).to_excel(writer, sheet_name='Ground-Truth', float_format='%.3f')

        # plot    
        row_name = [r'ESD (cm)', r'Shortest Dimension (cm)', r'Intermediate Dimension (cm)', r'Longest Dimension (cm)', r'$Volume\ (cm^3)$', r'$Area\ (cm^2)$', r'$FER_{3D}$', r'$Sphericity_{3D}$']
        plot_name = [ self.filestem + '_' + suffix for suffix in ['ESD.png', 'a.png', 'b.png', 'c.png', 'Volume.png', 'Area.png', 'FER3D.png', 'Sphericity3D.png'] ]
        plot_xlim = [(0,50), (0,50), (0,50), (0,50), (0,7000), (0,3000), (1,3), (0.5,1.0)]
        for i, name in enumerate(plot_name):
            fig = plt.figure()
            markersize = 3
            pass_point = np.min(gt_valid[i] * 0.95)
            plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')
            plt.plot(gt_valid[i], pred_valid[i], 'o', markerfacecolor='none', markersize=markersize)
            plt.gca().set_aspect('equal')

            mape = np.sum(np.abs(pred_valid[i] - gt_valid[i]) / gt_valid[i] * 100) / gt_valid[i].shape[0]
            plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

            plt.ylim(plot_xlim[i])
            plt.xlim(plot_xlim[i])
            plt.xlabel(row_name[i]+r', Ground-Truth')
            plt.ylabel(row_name[i]+r', Prediction')
            plt.grid()
            fig.savefig(os.path.join(self.output_path, name), bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close(fig)
        
        # bubble plot
        for i, name in enumerate(plot_name):
            fig = plt.figure()
            markersize = 3
            pass_point = np.min(gt_valid[i] * 0.95)
            plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')

            shape_percentage = [1000 ** x for x in pred_valid[8]]
            plt.scatter(gt_valid[i], pred_valid[i], c='royalblue', alpha=0.5, s=shape_percentage, zorder=2)
            for j, x in enumerate(pred_valid[8]):
                plt.text(gt_valid[i][j], pred_valid[i][j], str(int(x*100)), horizontalalignment='center', verticalalignment='center', fontsize='x-small')
            plt.gca().set_aspect('equal')

            # mape = np.sum(np.abs(pred_valid[i] - gt_valid[i]) / gt_valid[i] * 100) / gt_valid[i].shape[0]
            # plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

            # plt.ylim(plot_xlim[i])
            # plt.xlim(plot_xlim[i])
            plt.xlabel(row_name[i]+r', Ground-Truth')
            plt.ylabel(row_name[i]+r', Prediction')
            plt.grid()
            fig.savefig(os.path.join(self.output_path, os.path.splitext(name)[0]+'_bubble.png'), bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close(fig)

    def parse_spreadsheet_kankakee(self):
        # prediction
        pred_df = pd.read_excel(self.input_spreadsheet, sheet_name='Completion')
        pred = pred_df.iloc[:9,1:].dropna().to_numpy()

        # gt
        gt_rr3r = pd.read_excel(os.path.join(self.gt_path, 'RR3R.xlsx'), sheet_name='Field Stats').dropna().to_numpy()[:,1:]

        gt_rr4k = pd.read_excel(os.path.join(self.gt_path, 'RR4K.xlsx'), sheet_name='Field Stats').dropna().to_numpy()[:,1:]

        gt_rr5k = pd.read_excel(os.path.join(self.gt_path, 'RR5K.xlsx'), sheet_name='Field Stats').dropna().to_numpy()[:,1:]

        gt_map = {30: gt_rr3r, 40: gt_rr4k, 50: gt_rr5k}

        gt = np.ones_like(pred)
        # -1 means over segmentation, -2 means undersegmentation, 0 means no visible corresponce from images, ID is 4XX for RR4, 3XX for RR3, 40XX for RR4K, 50XX for RR5K
        gt_ids = pred_df.iloc[9, 1:].dropna().to_numpy().astype(np.int32)
        for ins_id, gt_id in enumerate(gt_ids):
            if gt_id > 0:
                cat, id = gt_id // 100, gt_id % 100
                gt[4, ins_id] = gt_map[cat][id-1,0] # volume in (cm^3)
 
        # only plot those rocks with valid GT correspondence & with success mesh reconstruction (failed reconstruction will save ESD, volume, area as 0)
        ins_mask = np.logical_and(gt_ids > 0, np.logical_and(pred[0] > 1e-4, pred[8] < 0.95)) # shape percentage > 0.95 usually is a planar
        num_oversegmentation = np.count_nonzero(gt_ids == -1)
        num_undersegmentation = np.count_nonzero(gt_ids == -2)
        num_gt_missing = np.count_nonzero(gt_ids == 0)

        pred_valid = pred[:, ins_mask]
        gt_valid = gt[:, ins_mask]
        print(f'{pred_valid.shape[1]} out of {pred.shape[1]} instances have valid ground-truth match. {num_oversegmentation} oversegmented instances, {num_undersegmentation} undersegmented instances, {num_gt_missing} ground-truth not visible.')

        # write to spreasheet
        save_spreadsheet_name = self.filestem + '_benchmark.xlsx'
        spreadsheet = os.path.join(self.output_path, save_spreadsheet_name)
        with pd.ExcelWriter(spreadsheet, mode='w') as writer:
            pd.DataFrame(pred_valid).to_excel(writer, sheet_name='Completion', float_format='%.3f')
            pd.DataFrame(gt_valid).to_excel(writer, sheet_name='Ground-Truth', float_format='%.3f')

        # convert from volume (cm^3) to weight (kg)
        unit_weight = 2.65 # g/cm^3
        pred_valid[4,:] = pred_valid[4,:] * unit_weight * 1e-3
        gt_valid[4,:] = gt_valid[4,:] * unit_weight * 1e-3

        # plot    
        row_name = [r'$Weight\ (kg)$']
        plot_name = [ self.filestem + '_' + suffix for suffix in ['Weight.png'] ]
        plot_xlim = [(0,100)]
        for i, name in enumerate(plot_name):
            fig = plt.figure()
            markersize = 3
            pass_point = np.min(gt_valid[4] * 0.95)
            plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')
            plt.plot(gt_valid[4], pred_valid[4], 'o', markerfacecolor='none', markersize=markersize)
            plt.gca().set_aspect('equal')

            mape = np.sum(np.abs(pred_valid[4] - gt_valid[4]) / gt_valid[4] * 100) / gt_valid[4].shape[0]
            plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

            plt.ylim(plot_xlim[i])
            plt.xlim(plot_xlim[i])
            plt.xlabel(row_name[i]+r', Ground-Truth')
            plt.ylabel(row_name[i]+r', Prediction')
            plt.grid()
            fig.savefig(os.path.join(self.output_path, name), bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close(fig)

        # bubble plot
        for i, name in enumerate(plot_name):
            fig = plt.figure()
            markersize = 3
            pass_point = np.min(gt_valid[i] * 0.95)
            plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')

            shape_percentage = [1000 ** x for x in pred_valid[8]]
            plt.scatter(gt_valid[4], pred_valid[4], c='royalblue', alpha=0.5, s=shape_percentage, zorder=2)
            for j, x in enumerate(pred_valid[8]):
                plt.text(gt_valid[4][j], pred_valid[4][j], str(int(x*100)), horizontalalignment='center', verticalalignment='center', fontsize='x-small')
            plt.gca().set_aspect('equal')

            # mape = np.sum(np.abs(pred_valid[i] - gt_valid[i]) / gt_valid[i] * 100) / gt_valid[i].shape[0]
            # plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

            # plt.ylim(plot_xlim[i])
            # plt.xlim(plot_xlim[i])
            plt.xlabel(row_name[i]+r', Ground-Truth')
            plt.ylabel(row_name[i]+r', Prediction')
            plt.grid()
            fig.savefig(os.path.join(self.output_path, os.path.splitext(name)[0]+'_bubble.png'), bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close(fig)

def plot_selected_stockpiles(fids, fig_prefix='rr3'):
    results_path = 'H:/AggregateStockpile/segmentation_results'
    folderlist = [folder for folder in os.listdir(results_path) if not os.path.splitext(folder)[1]]

    row_name = [r'ESD (cm)', r'Shortest Dimension (cm)', r'Intermediate Dimension (cm)', r'Longest Dimension (cm)', r'$Volume\ (cm^3)$', r'$Area\ (cm^2)$', r'$FER_{3D}$', r'$Sphericity_{3D}$']
    plot_name = [f'{fig_prefix}_ESD.png', f'{fig_prefix}_a.png', f'{fig_prefix}_b.png', f'{fig_prefix}_c.png', f'{fig_prefix}_Volume.png', f'{fig_prefix}_Area.png', f'{fig_prefix}_FER3D.png', f'{fig_prefix}_Sphericity3D.png']
    if fig_prefix == 'rr3':
        plot_xlim = [(0,20), (0,20), (0,20), (10,30), (0,1500), (0,1000), (1,3), (0.5,1.0)]
    else: 
        plot_xlim = [(0,50), (0,50), (0,50), (0,50), (0,7000), (0,3000), (1,3), (0.5,1.0)]
    markers = ['o', 's', 'v','^', 'X', 'D']
    colors = ['darkred', 'gold', 'royalblue', 'limegreen', 'darkorange', 'indigo']

    for i, name in enumerate(plot_name):
        fig = plt.figure()
        markersize = 4
        data_pred = np.empty(0)
        data_gt = np.empty(0)

        for j, fid in enumerate(fids):
            f = folderlist[fid]
            spreadsheet_path = os.path.join(results_path, f, f+'_benchmark.xlsx')
            
            fields = f.split('_')
            cat, stockpile_id = fields[0], fields[1]

            pred = pd.read_excel(spreadsheet_path, sheet_name='Completion').iloc[:,1:].dropna().to_numpy()
            gt = pd.read_excel(spreadsheet_path, sheet_name='Ground-Truth').iloc[:,1:].dropna().to_numpy()

            plt.plot(gt[i], pred[i], marker=markers[j], color=colors[j], linestyle='None', markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, label=cat+'-'+stockpile_id)
            data_pred = np.concatenate((data_pred, pred[i]))
            data_gt = np.concatenate((data_gt, gt[i]))

        # if 'Volume' in name:
        #     # plot correction factor line
        #     xdata, ydata = data_gt, data_pred
        #     # Method 2: force passing (x=0,y=0) 
        #     def fit_0(x, a):
        #         return a*(x-0) + 0
        #     popt, pcov = curve_fit(fit_0, xdata, ydata)
        #     # manually compute R^2: https://stackoverflow.com/a/37899817
        #     residuals = ydata- fit_0(xdata, *popt)
        #     ss_res = np.sum(residuals**2)
        #     ss_tot = np.sum((ydata-np.mean(ydata))**2)
        #     r_squared = 1 - (ss_res / ss_tot)

        #     ### plot regression line(s)
        #     xdata = np.append(xdata, 0) # extend the line
        #     plt.plot(xdata, fit_0(xdata, *popt),  'r-', linewidth=1, label='Correction Line')
        #     plt.gca().text(0.05,0.4,f'$y={popt[0]:.3f}\cdot x$\n'f'$R^2={r_squared:.2f}$', transform=plt.gca().transAxes, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black'), fontsize='x-small')

        pass_point = np.min(data_gt * 0.95)
        plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')
        plt.gca().set_aspect('equal')

        mape = np.sum(np.abs(data_pred - data_gt) / data_gt * 100) / data_gt.shape[0]
        plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

        plt.ylim(plot_xlim[i])
        plt.xlim(plot_xlim[i])
        plt.xlabel(row_name[i]+r', Ground-Truth')
        plt.ylabel(row_name[i]+r', Prediction')
        plt.grid()
        plt.legend(loc='upper left', fontsize='small')
        fig.savefig(os.path.join(results_path, name), bbox_inches='tight', dpi=300, transparent=True)
        # plt.show()
        plt.close(fig)


def plot_all_stockpiles(fids, fig_prefix='all'):
    results_path = 'H:/AggregateStockpile/segmentation_results'
    folderlist = [folder for folder in os.listdir(results_path) if not os.path.splitext(folder)[1]]

    row_name = [r'ESD (cm)', r'Shortest Dimension (cm)', r'Intermediate Dimension (cm)', r'Longest Dimension (cm)', r'$Volume\ (cm^3)$', r'$Area\ (cm^2)$', r'$FER_{3D}$', r'$Sphericity_{3D}$']
    plot_name = [f'{fig_prefix}_ESD.png', f'{fig_prefix}_a.png', f'{fig_prefix}_b.png', f'{fig_prefix}_c.png', f'{fig_prefix}_Volume.png', f'{fig_prefix}_Area.png', f'{fig_prefix}_FER3D.png', f'{fig_prefix}_Sphericity3D.png']
    plot_xlim = [(0,50), (0,50), (0,50), (0,50), (0,7000), (0,3000), (1,3), (0.5,1.0)]

    for i, name in enumerate(plot_name):
        fig = plt.figure()
        markersize = 3
        data_pred = np.empty(0)
        data_gt = np.empty(0)

        for fid in fids:
            f = folderlist[fid]
            spreadsheet_path = os.path.join(results_path, f, f+'_benchmark.xlsx')
            
            fields = f.split('_')
            cat, stockpile_id = fields[0], fields[1]

            pred = pd.read_excel(spreadsheet_path, sheet_name='Completion').iloc[:,1:].dropna().to_numpy()
            gt = pd.read_excel(spreadsheet_path, sheet_name='Ground-Truth').iloc[:,1:].dropna().to_numpy()

            marker = 'o' if cat == 'RR3' else 's'
            color = 'darkorange' if cat == 'RR3' else 'royalblue'
            plt.plot(gt[i], pred[i], linestyle='None', marker=marker, color=color, markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, label=cat+'-all')
            data_pred = np.concatenate((data_pred, pred[i]))
            data_gt = np.concatenate((data_gt, gt[i]))

        if 'Volume' in name:
            # plot correction factor line
            xdata, ydata = data_gt, data_pred
            # Method 2: force passing (x=0,y=0) 
            def fit_0(x, a):
                return a*(x-0) + 0
            popt, pcov = curve_fit(fit_0, xdata, ydata)
            # manually compute R^2: https://stackoverflow.com/a/37899817
            residuals = ydata- fit_0(xdata, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata-np.mean(ydata))**2)
            r_squared = 1 - (ss_res / ss_tot)

            ### plot regression line(s)
            # xdata = np.append(xdata, 0) # extend the line
            # plt.plot(xdata, fit_0(xdata, *popt),  'r-', linewidth=1, label='Correction Line')
            # plt.gca().text(0.05,0.6,f'$y={popt[0]:.3f}\cdot x$\n'f'$R^2={r_squared:.2f}$', transform=plt.gca().transAxes, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black'), fontsize='x-small')
            
        pass_point = np.min(data_gt * 0.95)
        plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')
        plt.gca().set_aspect('equal')

        mape = np.sum(np.abs(data_pred - data_gt) / data_gt * 100) / data_gt.shape[0]
        plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

        plt.ylim(plot_xlim[i])
        plt.xlim(plot_xlim[i])
        plt.xlabel(row_name[i]+r', Ground-Truth')
        plt.ylabel(row_name[i]+r', Prediction')
        plt.grid()

        handles, labels = plt.gca().get_legend_handles_labels() # for merging same labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')

        fig.savefig(os.path.join(results_path, name), bbox_inches='tight', dpi=300, transparent=True)
        # plt.show()
        plt.close(fig)

def plot_selected_kankakee_stockpiles(fids, fig_prefix='rr4k'):
    results_path = 'H:/AggregateStockpile/segmentation_results'
    folderlist = [folder for folder in os.listdir(results_path) if not os.path.splitext(folder)[1]]

    row_name = [r'$Weight\ (kg)$']
    plot_name = [f'{fig_prefix}_Weight.png']
    if fig_prefix == 'rr3r':
        plot_xlim = [(0,10)]
    else: 
        plot_xlim = [(0,80)]
    markers = ['o', 's', 'v','^', 'X', 'D']
    colors = ['darkred', 'gold', 'royalblue', 'limegreen', 'darkorange', 'indigo']

    for i, name in enumerate(plot_name):
        fig = plt.figure()
        markersize = 4
        data_pred = np.empty(0)
        data_gt = np.empty(0)

        for j, fid in enumerate(fids):
            f = folderlist[fid]
            spreadsheet_path = os.path.join(results_path, f, f+'_benchmark.xlsx')
            
            fields = f.split('_')
            cat, stockpile_id = fields[0], fields[1]

            pred = pd.read_excel(spreadsheet_path, sheet_name='Completion').iloc[:,1:].dropna().to_numpy()
            gt = pd.read_excel(spreadsheet_path, sheet_name='Ground-Truth').iloc[:,1:].dropna().to_numpy()

            # convert from volume (cm^3) to weight (kg)
            unit_weight = 2.65 # g/cm^3
            pred[4,:] = pred[4,:] * unit_weight * 1e-3
            gt[4,:] = gt[4,:] * unit_weight * 1e-3

            plt.plot(gt[4], pred[4], marker=markers[j], color=colors[j], linestyle='None', markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, label=cat+'-'+stockpile_id)
            data_pred = np.concatenate((data_pred, pred[4]))
            data_gt = np.concatenate((data_gt, gt[4]))

        # if 'Weight' in name:
        #     # plot correction factor line
        #     xdata, ydata = data_gt, data_pred
        #     # Method 2: force passing (x=0,y=0) 
        #     def fit_0(x, a):
        #         return a*(x-0) + 0
        #     popt, pcov = curve_fit(fit_0, xdata, ydata)
        #     # manually compute R^2: https://stackoverflow.com/a/37899817
        #     residuals = ydata- fit_0(xdata, *popt)
        #     ss_res = np.sum(residuals**2)
        #     ss_tot = np.sum((ydata-np.mean(ydata))**2)
        #     r_squared = 1 - (ss_res / ss_tot)

        #     ### plot regression line(s)
        #     xdata = np.append(xdata, 0) # extend the line
        #     plt.plot(xdata, fit_0(xdata, *popt),  'r-', linewidth=1, label='Correction Line')
        #     plt.gca().text(0.05,0.4,f'$y={popt[0]:.3f}\cdot x$\n'f'$R^2={r_squared:.2f}$', transform=plt.gca().transAxes, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black'), fontsize='x-small')

        pass_point = np.min(data_gt * 0.95)
        plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')
        plt.gca().set_aspect('equal')

        mape = np.sum(np.abs(data_pred - data_gt) / data_gt * 100) / data_gt.shape[0]
        plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

        plt.ylim(plot_xlim[i])
        plt.xlim(plot_xlim[i])
        plt.xlabel(row_name[i]+r', Ground-Truth')
        plt.ylabel(row_name[i]+r', Prediction')
        plt.grid()
        plt.legend(loc='upper left', fontsize='small')
        fig.savefig(os.path.join(results_path, name), bbox_inches='tight', dpi=300, transparent=True)
        # plt.show()
        plt.close(fig)


def plot_all_kankakee_stockpiles(fids, fig_prefix='field-all'):
    results_path = 'H:/AggregateStockpile/segmentation_results'
    folderlist = [folder for folder in os.listdir(results_path) if not os.path.splitext(folder)[1]]

    row_name = [r'$Weight\ (kg)$']
    plot_name = [f'{fig_prefix}_Weight.png']
    plot_xlim = [(0,80)]

    markers = {'RR3R': '^', 'RR4K': 'o', 'RR5K': 's'}
    colors = {'RR3R': 'r', 'RR4K': 'darkorange', 'RR5K': 'royalblue'}
    markersize = 3
    for i, name in enumerate(plot_name):
        fig = plt.figure()
        ax = plt.gca()
        data_pred = np.empty(0)
        data_gt = np.empty(0)
        data_pred_rr3 = np.empty(0)
        data_gt_rr3 = np.empty(0)

        for fid in fids:
            f = folderlist[fid]
            spreadsheet_path = os.path.join(results_path, f, f+'_benchmark.xlsx')
            
            fields = f.split('_')
            cat, stockpile_id = fields[0], fields[1]

            pred = pd.read_excel(spreadsheet_path, sheet_name='Completion').iloc[:,1:].dropna().to_numpy()
            gt = pd.read_excel(spreadsheet_path, sheet_name='Ground-Truth').iloc[:,1:].dropna().to_numpy()

            # convert from volume (cm^3) to weight (kg)
            unit_weight = 2.65 # g/cm^3
            pred[4,:] = pred[4,:] * unit_weight * 1e-3
            gt[4,:] = gt[4,:] * unit_weight * 1e-3

            marker = markers[cat]
            color = colors[cat]
            plt.plot(gt[4], pred[4], linestyle='None', marker=marker, color=color, markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, label=cat+'-all')
            data_pred = np.concatenate((data_pred, pred[4]))
            data_gt = np.concatenate((data_gt, gt[4]))
            if cat == 'RR3R':
                data_pred_rr3 = np.concatenate((data_pred_rr3, pred[4]))
                data_gt_rr3 = np.concatenate((data_gt_rr3, gt[4]))

        if 'Weight' in name:
            # plot correction factor line
            xdata, ydata = data_gt, data_pred
            # Method 2: force passing (x=0,y=0) 
            def fit_0(x, a):
                return a*(x-0) + 0
            popt, pcov = curve_fit(fit_0, xdata, ydata)
            # manually compute R^2: https://stackoverflow.com/a/37899817
            residuals = ydata- fit_0(xdata, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata-np.mean(ydata))**2)
            r_squared = 1 - (ss_res / ss_tot)

            ### plot regression line(s)
            # xdata = np.append(xdata, 0) # extend the line
            # plt.plot(xdata, fit_0(xdata, *popt),  'r-', linewidth=1, label='Correction Line')
            # plt.gca().text(0.05,0.6,f'$y={popt[0]:.3f}\cdot x$\n'f'$R^2={r_squared:.2f}$', transform=plt.gca().transAxes, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black'), fontsize='x-small')
            
        pass_point = np.min(data_gt * 0.95)
        plt.axline((pass_point, pass_point), slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line')
        plt.gca().set_aspect('equal')

        # zoomed inset plot for RR3R data
        # inset axes
        axins = ax.inset_axes([0.625, 0.125, 0.3125, 0.3125]) # lower-left corner + heigh width (in 0-1)
        axins.plot(data_gt_rr3, data_pred_rr3, linestyle='None', marker=markers['RR3R'], color=colors['RR3R'], markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, zorder=3)
        # sub region of the original image
        xymin, xymax = min(np.min(data_gt_rr3),np.min(data_pred_rr3))*0.95, max(np.max(data_gt_rr3),np.max(data_pred_rr3))*1.05
        axins.set_xlim(xymin, xymax)
        axins.set_ylim(xymin, xymax)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        pass_point = (0,0)
        axins.axline(pass_point, slope=1, linestyle='--', linewidth=1, color='k', label='Reference Line', zorder=2)
        # error_line_pass_point = (1,1) if 'FER3D' in name else (0,0) 
        # axins.axline(error_line_pass_point, slope=0.9, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        # axins.axline(error_line_pass_point, slope=1.1, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        # axins.axline(error_line_pass_point, slope=0.8, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2)
        # axins.axline(error_line_pass_point, slope=1.2, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2) # loosely dashes
        # axins.set_aspect('equal')
        ax.indicate_inset_zoom(axins, edgecolor="black") # auto zoom-in window

        mape = np.sum(np.abs(data_pred - data_gt) / data_gt * 100) / data_gt.shape[0]
        plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

        plt.ylim(plot_xlim[i])
        plt.xlim(plot_xlim[i])
        plt.xlabel(row_name[i]+r', Ground-Truth')
        plt.ylabel(row_name[i]+r', Prediction')
        plt.grid()

        handles, labels = plt.gca().get_legend_handles_labels() # for merging same labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')

        fig.savefig(os.path.join(results_path, name), bbox_inches='tight', dpi=300, transparent=True)
        # plt.show()
        plt.close(fig)

def plot_all_stockpiles_sp(fids, sp_threshold=0.75, fig_prefix='all'):
    results_path = 'H:/AggregateStockpile/segmentation_results'
    folderlist = [folder for folder in os.listdir(results_path) if not os.path.splitext(folder)[1]]

    row_name = [r'ESD (cm)', r'Shortest Dimension (cm)', r'Intermediate Dimension (cm)', r'Longest Dimension (cm)', r'$Volume\ (cm^3)$', r'$Area\ (cm^2)$', r'$FER_{3D}$', r'$Sphericity_{3D}$']
    fig_prefix = fig_prefix + '_sp' + str(int(sp_threshold*100))
    plot_name = [f'{fig_prefix}_ESD.png', f'{fig_prefix}_a.png', f'{fig_prefix}_b.png', f'{fig_prefix}_c.png', f'{fig_prefix}_Volume.png', f'{fig_prefix}_Area.png', f'{fig_prefix}_FER3D.png', f'{fig_prefix}_Sphericity3D.png']
    plot_xlim = [(0,50), (0,50), (0,50), (0,50), (0,7000), (0,3000), (1,3), (0.5,1.0)]

    for i, name in enumerate(plot_name):
        fig = plt.figure()
        markersize = 3
        data_pred = np.empty(0)
        data_gt = np.empty(0)

        for fid in fids:
            f = folderlist[fid]
            spreadsheet_path = os.path.join(results_path, f, f+'_benchmark.xlsx')
            
            fields = f.split('_')
            cat, stockpile_id = fields[0], fields[1]

            pred = pd.read_excel(spreadsheet_path, sheet_name='Completion').iloc[:,1:].dropna().to_numpy()
            gt = pd.read_excel(spreadsheet_path, sheet_name='Ground-Truth').iloc[:,1:].dropna().to_numpy()

            # shape percentage is the last row, use it to filter the results
            sp = pred[-1]
            sp_mask = sp >= sp_threshold 
            pred = pred[:, sp_mask]
            gt = gt[:, sp_mask]

            marker = 'o' if cat == 'RR3' else 's'
            color = 'darkorange' if cat == 'RR3' else 'royalblue'
            plt.plot(gt[i], pred[i], linestyle='None', marker=marker, color=color, markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, label=cat+'-all', zorder=3)
            data_pred = np.concatenate((data_pred, pred[i]))
            data_gt = np.concatenate((data_gt, gt[i]))

        if 'Volume' in name:
            # plot correction factor line
            xdata, ydata = data_gt, data_pred
            # Method 2: force passing (x=0,y=0) 
            def fit_0(x, a):
                return a*(x-0) + 0
            popt, pcov = curve_fit(fit_0, xdata, ydata)
            # manually compute R^2: https://stackoverflow.com/a/37899817
            residuals = ydata- fit_0(xdata, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata-np.mean(ydata))**2)
            r_squared = 1 - (ss_res / ss_tot)

            ### plot regression line(s)
            # xdata = np.append(xdata, 0) # extend the line
            # plt.plot(xdata, fit_0(xdata, *popt),  'r-', linewidth=1, label='Correction Line')
            # plt.gca().text(0.05,0.6,f'$y={popt[0]:.3f}\cdot x$\n'f'$R^2={r_squared:.2f}$', transform=plt.gca().transAxes, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black'), fontsize='x-small')
            
        pass_point = np.min(data_gt * 0.95)
        plt.axline((pass_point, pass_point), slope=1, linestyle='-', linewidth=1, color='k', label='Reference Line', zorder=2)
        error_line_pass_point = (1,1) if 'FER3D' in name else (0,0) 
        plt.axline(error_line_pass_point, slope=0.9, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        plt.axline(error_line_pass_point, slope=1.1, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        plt.axline(error_line_pass_point, slope=0.8, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2)
        plt.axline(error_line_pass_point, slope=1.2, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2) # loosely dashes
        plt.gca().set_aspect('equal')

        mape = np.sum(np.abs(data_pred - data_gt) / data_gt * 100) / data_gt.shape[0]
        plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

        plt.ylim(plot_xlim[i])
        plt.xlim(plot_xlim[i])
        plt.title(f'Shape Percentage (SP) Threshold = {str(int(sp_threshold*100))}%')
        plt.xlabel(row_name[i]+r', Ground-Truth')
        plt.ylabel(row_name[i]+r', Prediction')
        plt.grid()

        handles, labels = plt.gca().get_legend_handles_labels() # for merging same labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')

        fig.savefig(os.path.join(results_path, name), bbox_inches='tight', dpi=300, transparent=True)
        # plt.show()
        plt.close(fig)

def plot_all_kankakee_stockpiles_sp(fids, sp_threshold=0.75, fig_prefix='field-all'):
    results_path = 'H:/AggregateStockpile/segmentation_results'
    folderlist = [folder for folder in os.listdir(results_path) if not os.path.splitext(folder)[1]]

    row_name = [r'$Weight\ (kg)$']
    fig_prefix = fig_prefix + '_sp' + str(int(sp_threshold*100))
    plot_name = [f'{fig_prefix}_Weight.png']
    plot_xlim = [(0,80)]

    markers = {'RR3R': '^', 'RR4K': 'o', 'RR5K': 's'}
    colors = {'RR3R': 'r', 'RR4K': 'darkorange', 'RR5K': 'royalblue'}
    markersize = 3
    for i, name in enumerate(plot_name):
        fig = plt.figure()
        ax = plt.gca()
        data_pred = np.empty(0)
        data_gt = np.empty(0)
        data_pred_rr3 = np.empty(0)
        data_gt_rr3 = np.empty(0)

        for fid in fids:
            f = folderlist[fid]
            spreadsheet_path = os.path.join(results_path, f, f+'_benchmark.xlsx')
            
            fields = f.split('_')
            cat, stockpile_id = fields[0], fields[1]

            pred = pd.read_excel(spreadsheet_path, sheet_name='Completion').iloc[:,1:].dropna().to_numpy()
            gt = pd.read_excel(spreadsheet_path, sheet_name='Ground-Truth').iloc[:,1:].dropna().to_numpy()

            # convert from volume (cm^3) to weight (kg)
            unit_weight = 2.65 # g/cm^3
            pred[4,:] = pred[4,:] * unit_weight * 1e-3
            gt[4,:] = gt[4,:] * unit_weight * 1e-3

            # shape percentage is the last row, use it to filter the results
            sp = pred[-1]
            sp_mask = sp >= sp_threshold 
            pred = pred[:, sp_mask]
            gt = gt[:, sp_mask]

            marker = markers[cat]
            color = colors[cat]
            plt.plot(gt[4], pred[4], linestyle='None', marker=marker, color=color, markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, label=cat+'-all', zorder=3)
            data_pred = np.concatenate((data_pred, pred[4]))
            data_gt = np.concatenate((data_gt, gt[4]))
            if cat == 'RR3R':
                data_pred_rr3 = np.concatenate((data_pred_rr3, pred[4]))
                data_gt_rr3 = np.concatenate((data_gt_rr3, gt[4]))

        if 'Weight' in name:
            # plot correction factor line
            xdata, ydata = data_gt, data_pred
            # Method 2: force passing (x=0,y=0) 
            def fit_0(x, a):
                return a*(x-0) + 0
            popt, pcov = curve_fit(fit_0, xdata, ydata)
            # manually compute R^2: https://stackoverflow.com/a/37899817
            residuals = ydata- fit_0(xdata, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata-np.mean(ydata))**2)
            r_squared = 1 - (ss_res / ss_tot)

            ### plot regression line(s)
            # xdata = np.append(xdata, 0) # extend the line
            # plt.plot(xdata, fit_0(xdata, *popt),  'r-', linewidth=1, label='Correction Line')
            # plt.gca().text(0.05,0.6,f'$y={popt[0]:.3f}\cdot x$\n'f'$R^2={r_squared:.2f}$', transform=plt.gca().transAxes, verticalalignment='top', color='black', bbox=dict(facecolor='white', edgecolor='black'), fontsize='x-small')
            
        pass_point = np.min(data_gt * 0.95)
        plt.axline((pass_point, pass_point), slope=1, linestyle='-', linewidth=1, color='k', label='Reference Line', zorder=2)
        error_line_pass_point = (1,1) if 'FER3D' in name else (0,0) 
        plt.axline(error_line_pass_point, slope=0.9, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        plt.axline(error_line_pass_point, slope=1.1, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        plt.axline(error_line_pass_point, slope=0.8, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2)
        plt.axline(error_line_pass_point, slope=1.2, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2) # loosely dashes
        plt.gca().set_aspect('equal')

        # zoomed inset plot for RR3R data
        # inset axes
        axins = ax.inset_axes([0.625, 0.125, 0.3125, 0.3125]) # lower-left corner + heigh width (in 0-1)
        axins.plot(data_gt_rr3, data_pred_rr3, linestyle='None', marker=markers['RR3R'], color=colors['RR3R'], markeredgecolor='k', markeredgewidth=0.5, markersize=markersize, zorder=3)
        # sub region of the original image
        xymin, xymax = min(np.min(data_gt_rr3),np.min(data_pred_rr3))*0.95, max(np.max(data_gt_rr3),np.max(data_pred_rr3))*1.05
        axins.set_xlim(xymin, xymax)
        axins.set_ylim(xymin, xymax)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        pass_point = (0,0)
        axins.axline(pass_point, slope=1, linestyle='-', linewidth=1, color='k', label='Reference Line', zorder=2)
        error_line_pass_point = (1,1) if 'FER3D' in name else (0,0) 
        axins.axline(error_line_pass_point, slope=0.9, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        axins.axline(error_line_pass_point, slope=1.1, linestyle=(0,(5,5)), linewidth=1, color='g', alpha=0.5, label="10% Error Line", zorder=2)
        axins.axline(error_line_pass_point, slope=0.8, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2)
        axins.axline(error_line_pass_point, slope=1.2, linestyle=(0,(5,5)), linewidth=1, color='b', alpha=0.5, label="20% Error Line", zorder=2) # loosely dashes
        axins.set_aspect('equal')
        ax.indicate_inset_zoom(axins, edgecolor="black") # auto zoom-in window
    
        mape = np.sum(np.abs(data_pred - data_gt) / data_gt * 100) / data_gt.shape[0]
        plt.text(0.7,0.05,f'MAPE={mape:.1f}%', transform=plt.gca().transAxes, color='black', bbox=dict(facecolor='white', edgecolor='black'))

        plt.ylim(plot_xlim[i])
        plt.xlim(plot_xlim[i])
        plt.title(f'Shape Percentage (SP) Threshold = {str(int(sp_threshold*100))}%')
        plt.xlabel(row_name[i]+r', Ground-Truth')
        plt.ylabel(row_name[i]+r', Prediction')
        plt.grid()

        handles, labels = plt.gca().get_legend_handles_labels() # for merging same labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')

        fig.savefig(os.path.join(results_path, name), bbox_inches='tight', dpi=300, transparent=True)
        # plt.show()
        plt.close(fig)

if __name__ == '__main__':
    results_path = 'H:/AggregateStockpile/segmentation_results'
    gt_path = 'H:/RockScan'

    folderlist = [folder for folder in os.listdir(results_path) if not os.path.splitext(folder)[1]]

    # # # rr3
    # fids = [8,9,10,11,12,13]
    # plot_selected_stockpiles(fids, fig_prefix='rr3')

    # # # rr4
    # fids = [17,18,19,20,21,22]
    # plot_selected_stockpiles(fids, fig_prefix='rr4')

    # # # all
    # fids = [8,9,10,11,12,13, 17,18,19,20,21,22]
    # plot_all_stockpiles(fids, fig_prefix='all')

    # # rr3r
    # fids = [0,1,2]
    # plot_selected_kankakee_stockpiles(fids, fig_prefix='rr3r')

    # # rr4k
    # fids = [14,15,16]
    # plot_selected_kankakee_stockpiles(fids, fig_prefix='rr4k')

    # # rr5k
    # fids = [23,24,25]
    # plot_selected_kankakee_stockpiles(fids, fig_prefix='rr5k')

    # # field all
    fids = [0,1,2,14,15,16,23,24,25]
    plot_all_kankakee_stockpiles(fids, fig_prefix='field-all')

    ## plot shape percentage
    sp_threshold = 0.75
    # fids = [8,9,10,11,12,13, 17,18,19,20,21,22]
    # plot_all_stockpiles_sp(fids, sp_threshold=sp_threshold, fig_prefix='all') # all re-engineered stockpiles

    # fids = [0,1,2,14,15,16,23,24,25]
    # plot_all_kankakee_stockpiles_sp(fids, sp_threshold=sp_threshold, fig_prefix='field-all') # all field stockpiles

    # start_id = 0#0
    # end_id = 2#22
    # for fid in range(start_id, end_id + 1):
    #     f = folderlist[fid]
    #     result_folder = os.path.join(results_path, f)

    #     print(f'Analyzing file {f}')
    #     a = Analysis3D(input_spreadsheet=os.path.join(result_folder, f+'.xlsx'), gt_path=gt_path, output_path=result_folder)

    #     if fid in [3,4,5,6,7]:
    #         continue # RR3_RR4_Mix skip for now
    #     elif fid in [8,9,10,11,12,13, 17,18,19,20,21,22]: # RR3 and RR4 ATREL data
    #         a.parse_spreadsheet()
    #     elif fid in [0,1,2,14,15,16,23,24,25]: 
    #         # RR3, RR4 and RR5 rantoul/kankakee field data, no full ground-truth
    #         a.parse_spreadsheet_kankakee()
        
