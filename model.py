from transformers.models.gpt2.modeling_tf_gpt2 import (
    TFGPT2Model,
    TFGPT2LMHeadModel,
    TFGPT2MainLayer,
    TFSharedEmbeddings,
    TFBaseModelOutputWithPast,
    TFAttention,
    TFBlock,
    TFMLP,
    input_processing,
    shape_list,
)
import tensorflow as tf


class TFQueryLayerAttention(TFAttention):
    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        use_cache,
        output_attentions,
        training=False,
        query_hidden_state=None,
    ):
        x = self.c_attn(x)
        query = self.c_attn(query_hidden_state)
        key, value = tf.split(x, 2, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        # to cope with keras serialization
        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        attn_outputs = self._attn(
            query,
            key,
            value,
            attention_mask,
            head_mask,
            output_attentions,
            training=training,
        )
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class TFQueryLayer(TFBlock):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        self.ln_1 = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_1"
        )
        self.attn = TFQueryLayerAttention(nx, n_ctx, config, scale, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_2"
        )
        self.mlp = TFMLP(inner_dim, config, name="mlp")

    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        use_cache,
        output_attentions,
        training=False,
        query_hidden_state=None,
    ):
        a = self.ln_1(x)
        output_attn = self.attn(
            a,
            layer_past,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
            training=training,
            query_hidden_state=query_hidden_state,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class PanguMainLayer(TFGPT2MainLayer):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.tqe = TFSharedEmbeddings(
            config.vocab_size,
            config.hidden_size,
            initializer_range=config.initializer_range,
            name="top_query_embedding",
            dtype="float32",
        )
        self.tq_layer = TFQueryLayer(
            config.n_ctx, config, scale=True, name="top_query_layer"
        )

    def call(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
            inputs["input_ids"] = tf.reshape(inputs["input_ids"], [-1, input_shape[-1]])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["past"] is None:
            past_length = 0
            inputs["past"] = [None] * len(self.h)
        else:
            past_length = shape_list(inputs["past"][0][0])[-2]

        if inputs["position_ids"] is None:
            inputs["position_ids"] = tf.expand_dims(
                tf.range(past_length, input_shape[-1] + past_length), axis=0
            )

        if inputs["attention_mask"] is not None:
            attention_mask_shape = shape_list(inputs["attention_mask"])
            inputs["attention_mask"] = tf.reshape(
                inputs["attention_mask"],
                (attention_mask_shape[0], 1, 1, attention_mask_shape[1]),
            )

            one_cst = tf.constant(1.0)
            inputs["attention_mask"] = tf.cast(
                inputs["attention_mask"], dtype=one_cst.dtype
            )
            inputs["attention_mask"] = tf.multiply(
                tf.subtract(one_cst, inputs["attention_mask"]), tf.constant(-10000.0)
            )

        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.num_hidden_layers

        inputs["position_ids"] = tf.reshape(
            inputs["position_ids"], [-1, shape_list(inputs["position_ids"])[-1]]
        )

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.wte(inputs["input_ids"], mode="embedding")

        position_embeds = tf.gather(self.wpe, inputs["position_ids"])

        if inputs["token_type_ids"] is not None:
            inputs["token_type_ids"] = tf.reshape(
                inputs["token_type_ids"], [-1, shape_list(inputs["token_type_ids"])[-1]]
            )
            token_type_embeds = self.wte(inputs["token_type_ids"], mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        position_embeds = tf.cast(position_embeds, dtype=inputs["inputs_embeds"].dtype)
        token_type_embeds = tf.cast(
            token_type_embeds, dtype=inputs["inputs_embeds"].dtype
        )
        hidden_states = inputs["inputs_embeds"] + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=inputs["training"])

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = () if inputs["use_cache"] else None
        all_attentions = () if inputs["output_attentions"] else None
        all_hidden_states = () if inputs["output_hidden_states"] else None
        for i, (block, layer_past) in enumerate(zip(self.h, inputs["past"])):
            if inputs["output_hidden_states"]:
                all_hidden_states = all_hidden_states + (
                    tf.reshape(hidden_states, output_shape),
                )

            outputs = block(
                hidden_states,
                layer_past,
                inputs["attention_mask"],
                inputs["head_mask"][i],
                inputs["use_cache"],
                inputs["output_attentions"],
                training=inputs["training"],
            )

            hidden_states, present = outputs[:2]
            if inputs["use_cache"]:
                presents = presents + (present,)

            if inputs["output_attentions"]:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)

        top_query_hidden_states = self.tqe(inputs["position_ids"], mode="embedding")
        hidden_states, present = self.tq_layer(
            hidden_states,
            layer_past,
            inputs["attention_mask"],
            inputs["head_mask"][i],
            inputs["use_cache"],
            inputs["output_attentions"],
            training=inputs["training"],
            query_hidden_state=top_query_hidden_states,
        )

        hidden_states = tf.reshape(hidden_states, output_shape)
        # Add last hidden state
        if inputs["output_hidden_states"]:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if inputs["output_attentions"]:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = (
                input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            )
            all_attentions = tuple(
                tf.reshape(t, attention_output_shape) for t in all_attentions
            )

        if not inputs["return_dict"]:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None
            )

        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# TODO: add_start_docstrings
class TFPanGuAlphaModel(TFGPT2Model):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = PanguMainLayer(config, name="transformer")


# TODO: add_start_docstrings
class TFPanGuAlphaLMHeadModel(TFGPT2LMHeadModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = PanguMainLayer(config, name="transformer")
