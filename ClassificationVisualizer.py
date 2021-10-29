from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import altair as alt
import pandas as pd
import numpy as np

alt.data_transformers.disable_max_rows()

AVAILABLE_SCORE_FUNCTIONS = [
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    cohen_kappa_score, 
    matthews_corrcoef, 
    roc_auc_score
]

class ClassificationVisualizer:
    def __init__(self):
        pass
    
    def compute_total_cost(self, *, fp, fn, fp_cost, fn_cost):
        return fp * fp_cost + fn * fn_cost
    
    def search_optimal_threshold_by_cost(self, *, y_true, y_probs, fp_cost=-1, fn_cost=-1):
        # define thresholds
        thresholds = np.arange(0, 1, 0.001)
        costs = []
        for t in thresholds:
            y_pred = self.to_labels(y_probs=y_probs, threshold=t)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost = self.compute_total_cost(fp=fp, fn=fn, fp_cost=fp_cost, fn_cost=fn_cost)
            costs.append(cost)
        costs = np.array(costs)
        # get best threshold
        ix = np.argmax(costs)
        return {
            'Optimal threshold': thresholds[ix],
            'Optimal cost': costs[ix],
            'Base threshold': 0.5,
            'Base cost': costs[np.where(thresholds == 0.5)][0],
            'Thresholds': thresholds,
            'Costs': costs
        }
    
    def to_labels(self, *, y_probs, threshold):
        return (y_probs[:, 1] >= threshold).astype('int')

    def scores_evolution_over_thresholds(self, *, y_true, y_probs):
        thresholds = np.arange(0.001, 1, 0.001)
        data = []
        for f in AVAILABLE_SCORE_FUNCTIONS:
            for t in thresholds:
                if f.__name__ == 'precision_score':
                    score = f(y_test, (y_probs[:, 1] > t).astype(int), 
                              zero_division=0)
                    data.append({
                        'Metric': f.__name__, 
                        'Threshold': round(t, 5), 
                        'Score': round(score, 4)
                    })
                else:
                    score = f(y_test, (y_probs[:, 1] > t).astype(int))
                    data.append({
                        'Metric': f.__name__, 
                        'Threshold': round(t, 5), 
                        'Score': round(score, 4)
                    })
        return pd.DataFrame(data)
    
    def dataframe_to_heatmap(self, *, df, title, x_title, y_title, 
                  domain_min=0.0, domain_max=1.0, 
                  width=300, height=200):
        """
            Render dataframes (confusion matrix and classification report for example)
            df must have this format for example (confusion matrix) : 
            pd.DataFrame({
                x: [0, 0, 1, 1],
                y: [0, 1, 0, 1],
                z: [0.2, 0.7, 0.8, 0.3]
            })
        """
        x = []
        for index in df.index.to_list():
            x.append([index] * len(df.columns))
        y = df.columns.to_list() * len(df.index)

        source = pd.DataFrame({
            'x': np.array(x).ravel(),
            'y': y,
            'z': df.values.ravel()
        })

        # Heatmap
        heatmap = alt.Chart(
            source, 
            title=alt.TitleParams(
                text=title, 
                fontSize=16, 
                fontStyle='italic'
            )
        ).mark_rect().encode(
                x=alt.X(
                    'x:O',
                    title=x_title,
                    axis=alt.Axis(
                        labelAngle=0
                    )
                ),
                y=alt.Y(
                    'y:O', 
                    title=y_title
                ),
                color=alt.Color(
                    'z:Q', 
                    legend=None, 
                    scale=alt.Scale(
                        reverse=False,
                        scheme="greens",
                        domain=[
                            domain_min, 
                            domain_max
                        ]
                    )
                )
            ).properties(
                width=width, 
                height=height
            )

        # Annotations
        annotations = alt.Chart(source).mark_text(
            baseline='middle'
        ).encode(
            x='x:O',
            y='y:O',
            text='z:Q'
        )
        return heatmap + annotations
    
    def render_scores_evolution_over_thresholds_lines(self, *, scores_evolution_over_thresholds_df, width=300, height=200):
        selection = alt.selection_multi(
            fields=['Metric'], 
            bind='legend'
        )
        line = alt.Chart(
            scores_evolution_over_thresholds_df,
            title=alt.TitleParams(
                text="Scores by thresholds", 
                fontSize=16, 
                fontStyle='italic')
        ).mark_line().encode(
            x='Threshold:Q', 
            y=alt.X('Score:Q'), 
            color=alt.Color('Metric:N', legend=alt.Legend(orient='left')),
            opacity=alt.condition(
                selection, 
                alt.value(1), 
                alt.value(0.33)
            )
        ).properties(
            width=width, 
            height=height
        ).add_selection(
            selection
        )
        return line
    
    def render_scores_evolution_over_thresholds_heatmap(self, *, scores_evolution_over_thresholds_df, width=300, height=200):
        selection = alt.selection_multi(
            fields=['Metric'], 
            bind='legend'
        )
        heatmap = alt.Chart(
            scores_evolution_over_thresholds_df,
            title=alt.TitleParams(
                text="Heatmap of scores by thresholds", 
                fontSize=16, 
                fontStyle='italic')
        ).mark_rect().encode(
            x='Threshold:O',
            y='Metric:O',
            color=alt.Color(
                'Score:Q', 
                scale=alt.Scale(
                    scheme="redyellowgreen",
                    reverse=False
                )
            ),
            tooltip=[
                alt.Tooltip('Metric:O'),
                alt.Tooltip('Threshold:O'),
                alt.Tooltip('Score:Q')
            ]
        ).properties(
            width=width, 
            height=height
        )
        return heatmap.add_selection(
            selection
        )
    
    def render_classification_report(self, *, y_true, y_probs, threshold, extra_title=None, width=300, height=200):
        extra_title = "" if extra_title is None else f"{extra_title.title()} -"
        y_pred = self.to_labels(y_probs=y_probs, threshold=threshold)
        cr = classification_report(
            y_true=y_true, 
            y_pred=y_pred, 
            output_dict=True
        )
        df = pd.DataFrame(cr).drop(
            'support', 
            errors='ignore'
        ).drop(
            'accuracy', 
            axis=1, 
            errors='ignore'
        ).applymap(
            lambda x: round(x, 2) 
            if isinstance(x, float) 
            else x
        )
        return self.dataframe_to_heatmap(
            df=df,
            title=f"{extra_title} Classification Report (Threshold={threshold})", 
            x_title="Metric",
            y_title="Class",
            width=width, 
            height=height
        )

    def render_confusion_matrix(self, *, y_true, y_probs, threshold, extra_title=None, width=300, height=200):
        extra_title = "" if extra_title is None else f"{extra_title.title()} -"
        y_pred = self.to_labels(y_probs=y_probs, threshold=threshold)
        cm = confusion_matrix(
            y_true=y_true, 
            y_pred=y_pred, 
            normalize='true'
        )
        df = pd.DataFrame(
            cm
        ).applymap(
            lambda x: round(x, 2) 
            if isinstance(x, float) 
            else x
        )

        x = []
        for index in df.index.to_list():
            x.append([index] * len(df.columns))
        y = df.columns.to_list() * len(df.index)

        source = pd.DataFrame({
            'x': np.array(x).ravel(),
            'y': y,
            'z': df.values.ravel()
        })

        return self.dataframe_to_heatmap(
            df=df, 
            title=f"{extra_title} Confusion Matrix (Threshold={threshold})", 
            x_title="Predicted label",
            y_title="True label",
            width=width, 
            height=height
        )

    def render_precision_recall(self, *, y_true, y_probs, optimal_threshold, width=300, height=200):
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs[:, 1])
        auc_score = auc(recall, precision)
        line = alt.Chart(
            pd.DataFrame({'Recall': recall, 'Precision': precision}),
            title=alt.TitleParams(
                text=f"Precision Recall Curve (auc={auc_score:.2f})", 
                fontSize=16, 
                fontStyle='italic'
            )
        ).mark_line(
            interpolate='step-before'
        ).encode(
            x='Recall:Q', 
            y='Precision:Q'
        ).properties(
            width=width, 
            height=height
        )

        idx_thresh, idy_thresh = (recall[np.argmin(abs(thresholds - .5))], 
                                  precision[np.argmin(abs(thresholds - .5))])

        idx_thresh_2, idy_thresh_2 = (recall[np.argmin(abs(thresholds - optimal_threshold))], 
                                  precision[np.argmin(abs(thresholds - optimal_threshold))])

        point = alt.Chart(
            pd.DataFrame({'Recall': [idx_thresh, idx_thresh_2], 
                          'Precision': [idy_thresh, idy_thresh_2],
                          'Gmean': [f"Th=0.50, Recall={idx_thresh:.2f}, Prec={idy_thresh:.2f}", 
                                    f"Th={optimal_threshold:.2f}, Recall={idx_thresh_2:.2f}, Prec={idy_thresh_2:.2f}"]
                         })
        ).mark_circle(
            size=100
        ).encode(
            x='Recall:Q',
            y='Precision:Q',
            color=alt.Color('Gmean:O', 
                            legend=alt.Legend(orient='bottom'),
                            scale=alt.Scale(scheme='Set1')
                           )
        )

        return line + point

    def render_roc_auc(self, *, y_true, y_probs, optimal_threshold, width=300, height=200):
        fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
        auc_score = auc(fpr, tpr)
        line = alt.Chart(
            pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr}),
            title=alt.TitleParams(
                text=f"ROC Curve (auc={auc_score:.2f})", 
                fontSize=16, 
                fontStyle='italic'
            )
        ).mark_line(
            interpolate='linear'
        ).encode(
            x='False Positive Rate:Q', 
            y='True Positive Rate:Q'
        ).properties(
            width=width, 
            height=height
        )

        idx_thresh, idy_thresh = (fpr[np.argmin(abs(thresholds - .5))], 
                                  tpr[np.argmin(abs(thresholds - .5))])


        idx_thresh_2, idy_thresh_2 = (fpr[np.argmin(abs(thresholds - optimal_threshold))], 
                                  tpr[np.argmin(abs(thresholds - optimal_threshold))])

        point = alt.Chart(
            pd.DataFrame({'False Positive Rate': [idx_thresh, idx_thresh_2], 
                          'True Positive Rate': [idy_thresh, idy_thresh_2],
                          'Gmean': [f"Th=0.50, Fpr={idx_thresh:.2f}, Tpr={idy_thresh:.2f}", 
                                    f"Th={optimal_threshold:.2f}, Fpr={idx_thresh_2:.2f}, Tpr={idy_thresh_2:.2f}"]
                         })
        ).mark_circle(
            size=100
        ).encode(
            x='False Positive Rate:Q',
            y='True Positive Rate:Q',
            color=alt.Color('Gmean:O', 
                            legend=alt.Legend(orient='bottom'), 
                            scale=alt.Scale(scheme='Set1')
                           )
        )

        return line + point
    
    def render_cost_by_threshold(self, *, optimal_threshold_by_cost_df, width=300, height=200):
        rdf = pd.DataFrame({
            'Threshold': optimal_threshold_by_cost_df['Thresholds'],
            'Cost': optimal_threshold_by_cost_df['Costs']
        }).applymap(
            lambda x: round(x, 2) 
            if isinstance(x, float) 
            else x
        )

        line = alt.Chart(
            rdf, 
            title=alt.TitleParams(
                text="Cost by Threshold", 
                fontSize=16, 
                fontStyle='italic'
            )
        ).mark_line().encode(
            x='Threshold:Q',
            y='Cost:Q'
        )

        rule = alt.Chart(
            pd.DataFrame([
                {"Threshold": round(optimal_threshold_by_cost_df['Optimal threshold'], 4)}
            ])
        ).mark_rule(
            color='green'
        ).encode(
            x='Threshold:Q'
        )

        rule2 = alt.Chart(
            pd.DataFrame([{"Threshold": 0.5}])
        ).mark_rule(
            color='red'
        ).encode(
            x='Threshold:Q'
        )

        annotation = alt.Chart(
            pd.DataFrame([
                {
                    "Threshold": round(optimal_threshold_by_cost_df['Optimal threshold'], 4), 
                    "Cost": round(optimal_threshold_by_cost_df['Optimal cost'], 4)
                }
            ])
        ).mark_text(
            align='right',
            baseline='middle',
            fontSize = 12,
            dy = 20,
            dx = -5,
            color='green'
        ).transform_calculate(
            combined_text = "'Optimal cost=' + datum.Cost + ' (Th=' + datum.Threshold + ')'"
        ).encode(
            x='Threshold',
            y='Cost',
            text='combined_text:O'
        )

        annotation2 = alt.Chart(
            pd.DataFrame([
                {
                    "Threshold": round(optimal_threshold_by_cost_df['Base threshold'], 4), 
                    "Cost": round(optimal_threshold_by_cost_df['Base cost'], 4)
                }
            ])
        ).mark_text(
            align='right',
            baseline='middle',
            fontSize = 12,
            dy = 60,
            dx = -5,
            color='red'
        ).transform_calculate(
            combined_text = "'Base cost=' + datum.Cost + ' (Th=' + datum.Threshold + ')'"
        ).encode(
            x='Threshold',
            y='Cost',
            text='combined_text:O'
        )

        return (
            line + rule + rule2 + annotation + annotation2
        ).properties(
            width=width, 
            height=height
        )
        
    def render_scores_by_base_and_optimal_thresholds(self, *, scores_by_base_and_optimal_thresholds_df, width=300, height=200):
        heatmap = alt.Chart(
            scores_by_base_and_optimal_thresholds_df,
            title=alt.TitleParams(
                text="Scores", 
                fontSize=16, 
                fontStyle='italic'
            )
        ).mark_rect().encode(
            y='Metric:O', 
            x=alt.X(
                'Threshold:O', 
                axis=alt.Axis(labelAngle=0)), 
            color=alt.Color(
                'Score:Q', 
                scale=alt.Scale(
                    scheme='redyellowgreen', 
                    domain=[0.0, 1.0]
                )
            )
        ).properties(
            width=width, 
            height=height
        )

        annotations = alt.Chart(
            scores_by_base_and_optimal_thresholds_df
        ).mark_text(
            baseline='middle'
        ).encode(
            x='Threshold:O',
            y='Metric:O',
            text='Score:Q'
        )
        return heatmap + annotations
    
    def generate_report(self, *, y_true, y_probs, fp_cost=-1, fn_cost=-1):
        """Computations"""
        # optimal_threshold_by_cost
        optimal_threshold_by_cost_df = \
        self.search_optimal_threshold_by_cost(
            y_true=y_true, 
            y_probs=y_probs,
            fp_cost=fp_cost, 
            fn_cost=fn_cost
        )
        # Optimal threshold
        optimal_threshold = optimal_threshold_by_cost_df['Optimal threshold']
        # Optimal cost
        optimal_cost = optimal_threshold_by_cost_df['Optimal cost']
        
        # Scores evolution over thresholds
        scores_evolution_over_thresholds_df = \
        self.scores_evolution_over_thresholds(
            y_true=y_true, 
            y_probs=y_probs
        )
        
        scores_by_base_and_optimal_thresholds_df = \
        scores_evolution_over_thresholds_df[
            scores_evolution_over_thresholds_df[
                'Threshold'
            ].isin(
                [0.5, optimal_threshold]
            )
        ]
        
        """Charts generating"""
        scores_by_base_and_optimal_thresholds_chart = \
        self.render_scores_by_base_and_optimal_thresholds(
            scores_by_base_and_optimal_thresholds_df=scores_by_base_and_optimal_thresholds_df,
            width=400
        )
        
        scores_evolution_over_thresholds_line_chart = \
        self.render_scores_evolution_over_thresholds_lines(
            scores_evolution_over_thresholds_df=scores_evolution_over_thresholds_df,
            width=400
        )
        
        scores_evolution_over_thresholds_heatmap_chart = \
        self.render_scores_evolution_over_thresholds_heatmap(
            scores_evolution_over_thresholds_df=scores_evolution_over_thresholds_df,
            width=400
        )
        
        cost_by_threshold_chart = self.render_cost_by_threshold(
            optimal_threshold_by_cost_df=optimal_threshold_by_cost_df,
            width=400
        )

        classification_report_chart = \
        self.render_classification_report(
            y_true=y_true,
            y_probs=y_probs,
            threshold=0.5,
            width=400
        )

        classification_report_optimal_chart = \
        self.render_classification_report(
            y_true=y_true,
            y_probs=y_probs,
            threshold=optimal_threshold,
            width=400
        )

        confusion_matrix_chart = \
        self.render_confusion_matrix(
            y_true=y_true,
            y_probs=y_probs,
            threshold=0.5,
            width=400
        )

        confusion_matrix_optimal_chart = \
        self.render_confusion_matrix(
            y_true=y_true,
            y_probs=y_probs,
            threshold=optimal_threshold,
            width=400
        )

        precision_recall_chart = \
        self.render_precision_recall(
            y_true=y_true,
            y_probs=y_probs,
            optimal_threshold=optimal_threshold,
            width=400
        )

        roc_auc_chart = \
        self.render_roc_auc(
            y_true=y_true,
            y_probs=y_probs,
            optimal_threshold=optimal_threshold,
            width=400
        )
        
        return alt.vconcat(
            alt.hconcat(scores_evolution_over_thresholds_line_chart, cost_by_threshold_chart),
            alt.hconcat(precision_recall_chart, roc_auc_chart),
            alt.hconcat(scores_by_base_and_optimal_thresholds_chart, scores_evolution_over_thresholds_heatmap_chart),
            alt.hconcat(classification_report_chart, classification_report_optimal_chart),
            alt.hconcat(confusion_matrix_chart, confusion_matrix_optimal_chart),
            center=True
        )
        
