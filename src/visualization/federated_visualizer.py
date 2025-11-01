"""
Federated visualizer for federated learning specific visualizations
Handles training progress, client comparisons, and aggregated analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import our utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualization_utils import (
    apply_consistent_styling, save_plot_with_metadata, 
    get_color_palette, ensure_directory_exists
)

logger = logging.getLogger(__name__)

class FederatedVisualizer:
    """Generate federated learning specific visualizations"""
    
    def __init__(self):
        apply_consistent_styling()
        self.colors = get_color_palette()
    
    def plot_training_vs_test_accuracy(self, federated_history: Dict, 
                                     output_path: str, client_histories: Dict = None) -> str:
        """
        Create training vs validation accuracy plots showing per-round progression
        
        Args:
            federated_history: Dictionary containing federated training history
            output_path: Path to save the plot
            client_histories: Optional client-specific histories
            
        Returns:
            str: Path where plot was saved
        """
        try:
            # Extract data from federated history
            rounds = list(range(1, len(federated_history.get('train_accuracy', [])) + 1))
            train_acc = federated_history.get('train_accuracy', [])
            test_acc = federated_history.get('test_accuracy', [])
            
            if not rounds or not train_acc:
                logger.warning("No training data available for accuracy plot")
                return ""
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Global aggregated metrics
            axes[0].plot(rounds, train_acc, marker='o', linewidth=3, markersize=8, 
                        color='blue', label='Training Accuracy', markerfacecolor='lightblue')
            if test_acc:
                axes[0].plot(rounds, test_acc, marker='s', linewidth=3, markersize=8, 
                            color='red', label='Test Accuracy', markerfacecolor='lightcoral')
            
            axes[0].set_title('Global Federated Training vs Test Accuracy', 
                             fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Communication Round')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1)
            
            # Add value annotations for key points
            for i, (r, acc) in enumerate(zip(rounds, train_acc)):
                if i % max(1, len(rounds)//5) == 0:  # Annotate every 5th point or so
                    axes[0].annotate(f'{acc:.3f}', (r, acc), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=9)
            
            # Plot 2: Per-client progression (if available)
            if client_histories:
                for client_id, history in client_histories.items():
                    client_train = history.get('train_accuracy', [])
                    client_test = history.get('test_accuracy', [])
                    
                    if client_train:
                        color_idx = int(client_id.split('_')[-1]) % len(self.colors)
                        axes[1].plot(rounds[:len(client_train)], client_train, 
                                   marker='o', linewidth=2, markersize=6,
                                   color=self.colors[color_idx], alpha=0.7,
                                   label=f'{client_id} Train')
                        
                        if client_test:
                            axes[1].plot(rounds[:len(client_test)], client_test, 
                                       marker='s', linewidth=2, markersize=6,
                                       color=self.colors[color_idx], alpha=0.7, 
                                       linestyle='--', label=f'{client_id} Test')
            else:
                # If no client histories, show global data again with different styling
                axes[1].plot(rounds, train_acc, marker='o', linewidth=2, 
                           color='blue', alpha=0.8, label='Global Training')
                if test_acc:
                    axes[1].plot(rounds, test_acc, marker='s', linewidth=2, 
                               color='red', alpha=0.8, label='Global Test')
            
            axes[1].set_title('Per-Client Training Progress', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Communication Round')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': 'Training vs Test Accuracy',
                'num_rounds': len(rounds),
                'final_train_accuracy': train_acc[-1] if train_acc else 0,
                'final_test_accuracy': test_acc[-1] if test_acc else 0,
                'num_clients': len(client_histories) if client_histories else 0
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"Training vs test accuracy plot saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating training vs test accuracy plot: {e}")
            return ""
    
    def plot_training_vs_test_loss(self, federated_history: Dict, 
                                 output_path: str, client_histories: Dict = None) -> str:
        """
        Create training vs validation loss plots showing per-round progression
        
        Args:
            federated_history: Dictionary containing federated training history
            output_path: Path to save the plot
            client_histories: Optional client-specific histories
            
        Returns:
            str: Path where plot was saved
        """
        try:
            # Extract data from federated history
            rounds = list(range(1, len(federated_history.get('train_loss', [])) + 1))
            train_loss = federated_history.get('train_loss', [])
            test_loss = federated_history.get('test_loss', [])
            
            if not rounds or not train_loss:
                logger.warning("No training data available for loss plot")
                return ""
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Global aggregated metrics
            axes[0].plot(rounds, train_loss, marker='o', linewidth=3, markersize=8, 
                        color='red', label='Training Loss', markerfacecolor='lightcoral')
            if test_loss:
                axes[0].plot(rounds, test_loss, marker='s', linewidth=3, markersize=8, 
                            color='purple', label='Test Loss', markerfacecolor='plum')
            
            axes[0].set_title('Global Federated Training vs Test Loss', 
                             fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Communication Round')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Add value annotations for key points
            for i, (r, loss) in enumerate(zip(rounds, train_loss)):
                if i % max(1, len(rounds)//5) == 0:  # Annotate every 5th point or so
                    axes[0].annotate(f'{loss:.3f}', (r, loss), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=9)
            
            # Plot 2: Per-client progression (if available)
            if client_histories:
                for client_id, history in client_histories.items():
                    client_train = history.get('train_loss', [])
                    client_test = history.get('test_loss', [])
                    
                    if client_train:
                        color_idx = int(client_id.split('_')[-1]) % len(self.colors)
                        axes[1].plot(rounds[:len(client_train)], client_train, 
                                   marker='o', linewidth=2, markersize=6,
                                   color=self.colors[color_idx], alpha=0.7,
                                   label=f'{client_id} Train')
                        
                        if client_test:
                            axes[1].plot(rounds[:len(client_test)], client_test, 
                                       marker='s', linewidth=2, markersize=6,
                                       color=self.colors[color_idx], alpha=0.7, 
                                       linestyle='--', label=f'{client_id} Test')
            else:
                # If no client histories, show global data again with different styling
                axes[1].plot(rounds, train_loss, marker='o', linewidth=2, 
                           color='red', alpha=0.8, label='Global Training')
                if test_loss:
                    axes[1].plot(rounds, test_loss, marker='s', linewidth=2, 
                               color='purple', alpha=0.8, label='Global Test')
            
            axes[1].set_title('Per-Client Loss Progress', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Communication Round')
            axes[1].set_ylabel('Loss')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': 'Training vs Test Loss',
                'num_rounds': len(rounds),
                'final_train_loss': train_loss[-1] if train_loss else 0,
                'final_test_loss': test_loss[-1] if test_loss else 0,
                'num_clients': len(client_histories) if client_histories else 0
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"Training vs test loss plot saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating training vs test loss plot: {e}")
            return ""
    
    def plot_client_performance_comparison(self, client_metrics: Dict[str, Dict], 
                                         output_path: str) -> str:
        """
        Create client performance comparison visualization
        
        Args:
            client_metrics: Dictionary mapping client_id to metrics
            output_path: Path to save the plot
            
        Returns:
            str: Path where plot was saved
        """
        try:
            if not client_metrics:
                logger.warning("No client metrics available for comparison")
                return ""
            
            # Extract data
            client_ids = list(client_metrics.keys())
            accuracies = [client_metrics[cid].get('accuracy', 0) for cid in client_ids]
            precisions = [client_metrics[cid].get('precision', 0) for cid in client_ids]
            recalls = [client_metrics[cid].get('recall', 0) for cid in client_ids]
            f1_scores = [client_metrics[cid].get('f1_score', 0) for cid in client_ids]
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot accuracy comparison
            axes[0, 0].bar(client_ids, accuracies, color=self.colors[0], alpha=0.8)
            axes[0, 0].set_title('Accuracy by Client', fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot precision comparison
            axes[0, 1].bar(client_ids, precisions, color=self.colors[1], alpha=0.8)
            axes[0, 1].set_title('Precision by Client', fontweight='bold')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(precisions):
                axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot recall comparison
            axes[1, 0].bar(client_ids, recalls, color=self.colors[2], alpha=0.8)
            axes[1, 0].set_title('Recall by Client', fontweight='bold')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(recalls):
                axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot F1-score comparison
            axes[1, 1].bar(client_ids, f1_scores, color=self.colors[3], alpha=0.8)
            axes[1, 1].set_title('F1-Score by Client', fontweight='bold')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(f1_scores):
                axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            plt.suptitle('Client Performance Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': 'Client Performance Comparison',
                'client_ids': client_ids,
                'metrics': {
                    'accuracies': accuracies,
                    'precisions': precisions,
                    'recalls': recalls,
                    'f1_scores': f1_scores
                }
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"Client performance comparison saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating client performance comparison: {e}")
            return ""
    
    def plot_convergence_analysis(self, federated_history: Dict, output_path: str) -> str:
        """
        Create convergence analysis visualization
        
        Args:
            federated_history: Dictionary containing federated training history
            output_path: Path to save the plot
            
        Returns:
            str: Path where plot was saved
        """
        try:
            rounds = list(range(1, len(federated_history.get('test_accuracy', [])) + 1))
            accuracies = federated_history.get('test_accuracy', [])
            
            if len(rounds) < 3:
                logger.warning("Insufficient data for convergence analysis")
                return ""
            
            # Calculate moving average and convergence rate
            window_size = min(3, len(accuracies))
            moving_avg = []
            for i in range(len(accuracies)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(accuracies[start_idx:i+1]))
            
            convergence_rate = [abs(accuracies[i] - accuracies[i-1]) for i in range(1, len(accuracies))]
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot convergence analysis
            axes[0].plot(rounds, accuracies, 'b-', linewidth=2, label='Raw Accuracy', alpha=0.6)
            axes[0].plot(rounds, moving_avg, 'r-', linewidth=3, 
                        label=f'Moving Average (window={window_size})')
            axes[0].set_title('Convergence Analysis', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Communication Round')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot convergence rate
            axes[1].plot(rounds[1:], convergence_rate, 'g-', linewidth=2, marker='o', markersize=6)
            axes[1].set_title('Convergence Rate', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Communication Round')
            axes[1].set_ylabel('|Accuracy Change|')
            axes[1].grid(True, alpha=0.3)
            
            # Add convergence threshold line
            threshold = 0.01
            axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                           label=f'Convergence Threshold ({threshold:.2f})')
            axes[1].legend()
            
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': 'Convergence Analysis',
                'num_rounds': len(rounds),
                'final_accuracy': accuracies[-1] if accuracies else 0,
                'convergence_rate': convergence_rate[-1] if convergence_rate else 0,
                'converged': convergence_rate[-1] < threshold if convergence_rate else False
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"Convergence analysis saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating convergence analysis: {e}")
            return ""
class E
nhancedFederatedVisualizer:
    """Main class integrating all visualization components for federated learning"""
    
    def __initsereturn Fal          }")
  on data: {eictiting predError validarror(f".eogger       l    ion as e:
 ept Except   exc     
     
       urn True    ret  
                 
 urn False       ret         ")
g}r_ms: {erroation faileddata validrediction "Prning(f logger.wa              lid:
 if not is_va                    

    d_proba'))('y_pregetpred_data.       
         , ['y_pred']pred_data_true'], ta['y     pred_da      
     a(atiction_dalidate_predor.vculats_cal self.metricerror_msg =valid,     is_   lator
     s calcuh metricwitdate  # Vali     
             alse
     urn F         ret           y}")
y: {ked keequiressing rning(f"Miogger.war       l           ne:
  a[key] is No or pred_datpred_data in  not key       if         :
keysuired_eqy in r      for ke  
      
          ')probaend('y_pred_s.apprequired_key            oba:
     require_pr        if     'y_pred']
= ['y_true',ired_keys        requ     
           False
  turn       re        ed_data:
  pr     if not   y:
     tr
       ""d content"tructure ann data sictioalidate pred """V  ol:
     ) -> bo False: bool =_probaire requ: Dict,pred_datata(self, n_date_predictiof _valida 
    de}
   eturn {      r")
      s plots: {e} progresingating train"Error gener.error(fger         loge:
    as ionxcept Except 
        e       
    turn files    re")
         filesng progressrainien(files)} trated {lGeneinfo(f"    logger.       
          ed_path
   is"] = savalysnvergence_an files["co          ath:
     ed_pav       if s
           h)
      utput_patory, oderated_hist      fe
          ysis(nalnce_at_converger.ploisualizeelf.fed_vved_path = s         sa
            ng")
   nalysis.pce_avergen       "con                            s", 
  resprogning_"trait_base, .join(outpu.patht_path = osutpu        o   ysis
  analrgencenerate conve        # Ge     
          aved_path
 ison"] = snce_comparmaperforent_s["cliile           f      :
       ved_pathif sa                
                  th)
       output_pat_metrics, clien                     ison(
  parcomrmance_perfoclient_alizer.plot_d_visuself.fe_path =        saved                  
             .png")
  _comparisonncermant_perfoie   "cl                                    
      rogress", g_p, "traininput_baseoutin(th.jos.pah = oput_patut  o                  rics:
ent_met if cli                       
 
       '))robad_p.get('y_pre_data      pred                     ed'], 
 a['y_pr, pred_dattrue']red_data['y_     p                     cs(
  n_metriificatiote_classculalator.calalcu_cics self.metr] =[client_idtricst_me      clien              :
    a)ed_datta(prn_daredictiolidate_pelf._va s   if              
   s.items():ictionent_predcliin data pred_nt_id, clier           fo= {}
      s trict_me    clien         
   tions:iclient_predf c        iison
    mparance cormperfot ienate cl    # Gener       
             th
 = saved_pat_loss"]ng_vs_tes["traini     files         
  ved_path:if sa              

          histories)ath, client_tput_pstory, oufederated_hi              s(
  _los_testning_vsr.plot_traiize_visualedlf.fed_path = se      sav           
  ")
     _loss.pngng_vs_test  "traini                                   ress", 
aining_progtrbase, "join(output_ = os.path.t_pathoutpu         ot
   plest loss aining vs trate trne    # Ge          
   h
        saved_pat"] =st_accuracytetraining_vs_iles["  f            _path:
      if saved  
             ies)
     _histor clientut_path,, outpated_history      feder          cy(
st_accurateg_vs_t_traininplolizer.ed_visualf.fath = se  saved_p       
              g")
 ccuracy.pn_test_a_vstraining         "                  
          ress", g_prog "traininut_base,utp.path.join(o = os_pathoutput        plot
    ccuracy  test ang vstraini # Generate             
                 }
           [])
    ss',t_loget('tesory.erated_histss': fedt_lo   'tes               ),
      []ain_loss', et('tr_history.gderatedloss': fein_     'tra            ,
       cy', [])st_accura('tey.getted_historfederacuracy':    'test_ac                  
   acy', []),curt('train_acy.georated_histacy': federaccur 'train_                    id] = {
   s[client_t_historieclien                     logs
iningctual tra ame fromese would coth          #          , 
 entational implem in rew -for nories toient hisclk    # Moc          
       ():ions.keysnt_predict_id in clieientcl    for      s:
       dictionient_pref cl        i
    = {}istories ient_h          clble
   if availaoriesistct client h  # Extra        try:
     
         }
    = {     files    ""
ions"sualizatgress viraining proe tGenerat """:
       [str, str]Dict -> ne)t = Nos: Diconredicti    client_p                                   : str,
utput_base: Dict, od_historyatefederlots(self, ess_ping_progrtrainrate_f gene
    de
    rn {}       retu
     es: {e}")ll curvcaprecision-reating neror ge"Errerror(f logger.     e:
       as t Exceptionexcep  
                  turn files
 re         )
  files"rve l cucalprecision-re(files)} ted {lenGenera(f"nfor.i       logge   
     
         _pathd"] = savede_aggregate"pr_curv      files[              path:
saved_        if 
               
         '))e_precision'averag.get(onsdictipreated_reg  agg                ath,
  utput_pve", o-Recall Cured Precision "Aggregat                 proba'],
  ['y_pred__predictionsaggregatedrue'], ctions['y_t_predigated   aggre          (
       rvell_cuion_recaplot_precisisualizer..eval_vath = self     saved_p
                      ng")
     _pr_curve.pted"aggrega                                   
       rves",ecall_cuecision_r "prutput_base,n(opath.joios.ath =   output_p             =True):
 obare_pr requiictions,ted_predaggregaon_data(ate_predictiself._validns and edictioegated_pr    if aggr    rve
    all cusion-reccireaggregated penerate        # G    
   
           saved_pathnt_id}"] =urve_{clie"pr_c files[f            
           ed_path:if sav              
                     n'))
     age_precisioa.get('aver    pred_dat                  
  utput_path, o",id}e - {client_urvRecall Ccision-     f"Pre                   
'],robay_pred_p['data pred_a['y_true'],d_dat     pre                  
 all_curve(ision_rect_precr.plozeali.eval_visu_path = self    saved               
              
       ve.png")_curid}_pr f"{client_                                    
         es",urvall_c_rec"precisione, (output_bas.joinpathh = os.ut_pat outp                  rue):
 proba=T require_ta,data(pred_darediction_._validate_pf self         i  :
     s()ictions.itement_preda in cli pred_datid,ent_  for cli           curves
-recallrecisiont per-clienate p     # Genery:
       tr      
         s = {}
 file
        "results""aggregated lients and or all cs fll curven-recaisioecrate pr"Gene""  r]:
      Dict[str, st -> : str)utput_bases: Dict, oiondictated_pre      aggreg                             , 
    tr, Dict]Dict[s: ionsent_predictlif, c_curves(selallrecsion_te_preci generaef
    
    dturn {}  re       e}")
   C curves: {g ROratin"Error geneer.error(flogg            e:
 on astit Excepep    exc
                rn files
     retu
       e files")ROC curvles)} fiated {len(o(f"Genernf    logger.i              
ath
      ] = saved_pggregated"roc_curve_ales["          fi         
  saved_path:          if  
                    auc'))
.get('roc_nspredictio aggregated_                 th,
   output_paC Curve", ROegated "Aggr               ba'],
    _proions['y_predictated_predgregtrue'], agtions['y_gated_predic aggre          (
         _curvezer.plot_rocsualielf.eval_vi = s saved_path                    
          png")
 _curve.ted_rocegagr      "ag                             
       ves",oc_curase, "rin(output_b= os.path.joth  output_pa        e):
       Truoba=equire_pr, rdictionsed_prea(aggregation_datictdate_predli self._vadictions andreated_preggg       if a    urve
 C cggregated ROrate a# Gene               
       _path
   = savedid}"]ent_rve_{cli"roc_cu    files[f           :
         pathaved_     if s         
                          auc'))
roc_d_data.get(' pre                     ath,
   output_p}",ent_id - {cli"ROC Curve     f             
      ba'],rored_p_data['y_ptrue'], preda['y_dat     pred_                 ve(
  urplot_roc_cvisualizer..eval_elf sh =saved_pat               
                   )
      g"e.pn_curvd}_roc"{client_i    f                                     
    ", urvesse, "roc_c(output_ba.joinpaths.= otput_path   ou               ue):
   a=Trprob require_red_data,a(pion_datredict._validate_pf self           i
     ):ms(ons.iteictint_predclieta in _dapredt_id, clien     for        es
curv ROC per-clientGenerate  # 
           y:
        tr         = {}
  files      ""
lts"ed resund aggregatclients afor all ves ROC-AUC curate "Gener  ""tr]:
       Dict[str, sstr) ->_base: ict, outputictions: Dated_pred   aggreg                       t], 
tr, Dic Dict[sdictions:re_pents(self, clirvecu_roc_ate def gener   
    {}
    return      ")
   }s: {etion report classifica generatingErrorr(f"logger.erro            
ption as e:t Exce  excep     
             les
urn fi        ret
    rt files")on repocatilassifis)} cen(fileerated {lfo(f"Gengger.in       lo      
        h
   ved_pat = saregated"]aggon_report_icati"classifiles[     f            h:
   d_patve       if sa    
                ")
     tReporification gated Class"Aggreitle=put_path, tout                    pred'], 
['y_dictionsed_preatregaggue'], ons['y_trdictireed_pgregatag            (
        ation_reportve_classificzer.sasualif.eval_vith = selsaved_pa          
                   
   port.txt")_reficationlassiregated_c       "agg                                  s", 
rtepoion_rclassificat"put_base, join(out = os.path.output_path       
          file  # Save to           
                
   )n Report"catiofied Classigatgrele="Ag         tit        d'],
   _preictions['yed_predegat aggrue'],'y_trctions[prediaggregated_                  
  (rtepotion_rficaint_classilizer.pruavis  self.eval_            e
   consolnt to     # Pri        :
   ions)predictegated_aggr_data(predictionlidate_._vaselfnd ns actioprediaggregated_f         iort
    ication repted classife aggrega  # Generat          
            ath
saved_p] = t_id}"{clienn_report_icatio"classiffiles[f                 :
       _pathaved    if s       
                          )
   ent_id}"eport - {cliication R"Classif    title=f             ,
        output_path_pred'],['yd_dataue'], prea['y_trred_dat p                    eport(
   ification_rclasssave_visualizer.self.eval__path = ed  sav          
                           ")
 _report.txticationsifd}_clasent_i{cli      f"                                      orts", 
 fication_repse, "classibautput_path.join(oh = os. output_pat                 
  e fil   # Save to              
                 
      ")client_id}- {tion Report ificale=f"Class         tit         
       red'],_data['y_pred_true'], pdata['y    pred_                    on_report(
tificalassi.print_c_visualizer self.eval             sole
      connt to  # Pri           ):
        _dataredction_data(pate_predielf._valid        if s
        ns.items():ctio_predientclita in red_daient_id, por cl  f          on reports
classificatint er-clieenerate p        # G
    try:
                s = {}
    file"
    ""ed resultsgatd aggrents anlie all crts forrepon icatioifrate classne"""Ge    
    str, str]:tr) -> Dict[_base: sDict, outputons: _predictigregated      ag                              ict], 
  ict[str, Dns: Ddictio, client_preself_reports(assificationenerate_cl   def g{}
    
    return      {e}")
    s: ceion matriting confusror genera.error(f"Er logger            as e:
ceptionxcept Ex       e         
 es
     return fil         ")
  filesrixatusion miles)} conf {len(fedeneratf"Ggger.info(        lo 
              h
  = saved_pat"]aggregatedion_matrix_es["confus  fil             path:
     if saved_             
                ys()))
   ictions.kepredst(client_ut_path, licms, outp client_                   _matrix(
sioned_confuot_aggregatlizer.pleval_visualf.th = sesaved_pa                
                ")
matrix.pngsion_nfuated_co"aggreg                                         trices", 
nfusion_mat_base, "coh.join(outpupat= os.ath output_p          
      tions):ed_predicgatgredata(agn_te_predictiolidaself._vaictions and redted_pand aggregant_cms      if clieix
       nfusion matrgregated corate ag# Gene     
             
      , [0, 0]]))), 0][0   [                                                           
  ix',_matr'confusion_data.get(rray(predappend(np.ams.ient_c    cl              th
      d_pa"] = saveid}client_ix_{n_matr[f"confusiofiles                      ath:
  ed_p      if sav              
                   th)
 utput_pa", o{client_id}Matrix - Confusion        f"           
      ],d'y_prea[''], pred_datrue_data['y_t     pred               trix(
    maonfusion_izer.plot_cisualeval_vf.= sel_path aved       s                   
            
  rix.png")ion_matid}_confus{client_       f"                                  s", 
    on_matrice"confusiutput_base, in(oh.joath = os.patput_pout                  data):
  d_ata(prerediction_didate_pself._val   if             .items():
 ions_predictin clientpred_data d, client_ifor           = []
  _cms  client    es
       ion matricient confuserate per-cl   # Gen   ry:
          t    
        
files = {}       "
 ""d resultsaggregatelients and for all ctrices ion maate confus""Gener      "r]:
  , stt[strDic) -> strtput_base: ct, ouions: Dited_predictega  aggr                        
        ], [str, Dictions: Dictt_predictienlf, clrices(senfusion_matnerate_co  def ge{}
    
    return         e}")
  ons: {isualizati v all generatingorErrr.error(f"ge        log
    n as e:xceptiopt Exce    e       
        d_files
 urn generate       ret   
  base}")put_s in {outn fileiozatvisualiles)} ted_fi {len(generaf"Generateder.info(   logg 
               s)
     g_file(traininteupdarated_files.        gene    )
predictionslient_ase, coutput_bhistory, ated_ feder           plots(
    s_progreste_training_lf.genera_files = seraining       t    ss plots
 re progte trainingnera. Ge     # 5
             les)
      pr_fie(s.updatted_fileenera    ge)
        , output_baspredictionsggregated_s, a_prediction      client          ves(
_cur_recallecisionprlf.generate_les = se   pr_fi        urves
 ll crecaecision-rate pr  # 4. Gene
                     files)
 date(roc_ted_files.upra        genese)
    ut_baons, outpdicti_preaggregatedictions, nt_pred       clie  (
       e_roc_curvesratlf.gene sees =    roc_fil        s
C curveenerate RO  # 3. G
                      es)
_filpdate(reportes.uerated_fil gen        ase)
   s, output_b_prediction, aggregateddictions client_pre          (
     reportsification_e_clasself.generatles = s_fireport          eports
  ication rssifate cla  # 2. Gener            
          s)
n_fileate(confusiod_files.upd  generate         ut_base)
 ns, outpctiodid_preggregatedictions, apre client_               es(
n_matricusionfcote_lf.genera_files = se   confusion         es
ric matfusiononenerate c# 1. G         
           }
    = {ated_files    gener
                 
    tamp=True)r, times_dioutputory(ect_dirutput create_oe =_bastput  ou
          ctoryut direzed outprgani  # Create o            
         
 directoryoutput_port create_s imtion_utilsualiza.viils   from ..ut
            try:  """
      aths
     le pype to fion talizatimapping visuDict            
 rns:   Retu    
     y
        ut directoroutpe asut_dir: B       outp
     oryng histniaied try: Federatted_histor  federa       
   on datapredictiAggregated ctions: egated_predigr     ag
       tion datapredicient_id to pping cls: Dict maictionedient_pr     cl        Args:
      
    
     tionsualizarated visede enhanced fting allr generat fory poinMain ent"
             ""]:
   ict[str, str Dions") ->idatults/valesstr = "rr: utput_di     o                    
         y: Dict,ted_historedera           f                
       s: Dict,ionted_predict aggrega                           ], 
      t[str, Dicticons: Dredictilient_p cself,alizations(suall_vienerate_ 
    def g)
   lized"er initiaVisualizrated d Fedence"Enhafo(inr.     logge  
   
      or()lattricsCalcu= Meator culetrics_cal   self.m
     izer()sualeratedVizer = Fedaliisu self.fed_v    zer()
   tionVisualialuaualizer = Evelf.eval_vis        s
        
oricsCalculatport Metrr imcalculatocs_n.metrialuatio   from ..evlizer
     onVisuaaluatiimport Evalizer on_visuuatial from .ev    
   elf):__(s