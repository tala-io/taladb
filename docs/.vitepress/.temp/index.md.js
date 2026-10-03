import { resolveComponent, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "./plugin-vue_export-helper.1tPrXgE0.js";
const __pageData = JSON.parse('{"title":"TalaDB — An open-source embedded vector and document database for building local-first AI applications.","description":"Store documents, metadata, and vectors together. Query structured data and semantic similarity from one embedded database — across the browser, Node.js, and React Native. No cloud required.","frontmatter":{"layout":"home","title":"TalaDB — An open-source embedded vector and document database for building local-first AI applications.","description":"Store documents, metadata, and vectors together. Query structured data and semantic similarity from one embedded database — across the browser, Node.js, and React Native. No cloud required."},"headers":[],"relativePath":"index.md","filePath":"index.md"}');
const _sfc_main = { name: "index.md" };
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  const _component_HomeHero = resolveComponent("HomeHero");
  const _component_HomeSocialProof = resolveComponent("HomeSocialProof");
  const _component_HomeFeatures = resolveComponent("HomeFeatures");
  const _component_HomeComparison = resolveComponent("HomeComparison");
  const _component_HomeUseCases = resolveComponent("HomeUseCases");
  const _component_HomeQuickStart = resolveComponent("HomeQuickStart");
  const _component_HomeCTA = resolveComponent("HomeCTA");
  _push(`<div${ssrRenderAttrs(_attrs)}>`);
  _push(ssrRenderComponent(_component_HomeHero, null, null, _parent));
  _push(ssrRenderComponent(_component_HomeSocialProof, null, null, _parent));
  _push(ssrRenderComponent(_component_HomeFeatures, null, null, _parent));
  _push(ssrRenderComponent(_component_HomeComparison, null, null, _parent));
  _push(ssrRenderComponent(_component_HomeUseCases, null, null, _parent));
  _push(ssrRenderComponent(_component_HomeQuickStart, null, null, _parent));
  _push(ssrRenderComponent(_component_HomeCTA, null, null, _parent));
  _push(`</div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("index.md");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender]]);
export {
  __pageData,
  index as default
};
