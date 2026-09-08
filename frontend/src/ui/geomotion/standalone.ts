export const GEOMOTION_STANDALONE_PARAM = "geomotion";

export function isGeoMotionStandalone(search = window.location.search) {
  return new URLSearchParams(search).get("view") === GEOMOTION_STANDALONE_PARAM;
}

export function geoMotionStandaloneUrl(location = window.location) {
  const url = new URL(location.href);
  url.searchParams.set("view", GEOMOTION_STANDALONE_PARAM);
  url.hash = "";
  return url.toString();
}
