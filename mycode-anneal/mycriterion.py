from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
import logging
import torch
import math
from fairseq import metrics, utils

@register_criterion('my_label_smoothed_cross_entropy')
class MyLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        mask_loss_weight,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.mask_loss_weight=mask_loss_weight


    def add_args(parser):
        parser.add_argument('--mask-loss-weight', default=0., type=float,
                            help='weight of mask loss')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        
    

    def forward(self, model, sample, reduce=True,show=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

            # return super().forward(model, sample, reduce=reduce)
        net_output, net_output_mask = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        src_len=net_output[-1]["mask"][0].size()[-1]
        mask_ave = net_output[-1]["mask"][0].mean(dim=0).mean(dim=0).mean(dim=-1).sum()
        
        mask_loss, _ = self.compute_loss(model, net_output_mask, sample, reduce=reduce)
        p_norm = torch.norm(1-net_output[-1]["mask"][0], p=2)/src_len
        mask_loss_final = -mask_loss+self.mask_loss_weight*p_norm

        logging_output = {
            "loss": loss.data,
            "mask_loss": mask_loss.data,
            "p2":p_norm.data,
            "nll_loss": nll_loss.data,
            'mask_ave': mask_ave.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "new_weight": model.get_weight(),
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        del mask_ave, nll_loss,mask_loss,p_norm
        return loss, mask_loss_final, sample_size, logging_output


    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        
        mask_loss_sum = sum(log.get('mask_loss', 0) for log in logging_outputs)
        # mask_loss_final_sum = sum(log.get('mask_loss_final', 0) for log in logging_outputs)
        p_sum = sum(log.get('p2', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        mask_sum = sum(log.get('mask_ave', 0) for log in logging_outputs) 

        metrics.log_scalar('mask_loss', mask_loss_sum / sample_size / math.log(2), sample_size, round=6)
        # metrics.log_scalar('mask_loss_final', mask_loss_final_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('p_2', p_sum / sample_size, sample_size, round=5)
        metrics.log_scalar('mask_ave', mask_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('new_weight', logging_outputs[0].get("new_weight",0)/4, len(logging_outputs), round=3)