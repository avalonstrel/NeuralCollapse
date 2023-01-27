

def display_metrics_dict(metrics_names, metrics_dict):
    metrics_messages = []
    for name in metrics_names:
        metric = metrics_dict[name]
        if isinstance(metric, float):
            metrics_messages.append('{}:{:.2f}'.format(name, metric))
        else:
            metrics_messages.append('{}:{}'.format(name, metric))

    metrics_message = ', '.join(metrics_messages)
    return metrics_message    



__all__ = ['display_metrics_dict']