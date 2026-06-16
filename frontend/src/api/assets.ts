import type { ApiList, AssetRecord } from "../types";
import { apiRequest } from "./client";

export const listAssets = () => apiRequest<ApiList<AssetRecord>>("/api/assets");
export const getAsset = (id: string) =>
  apiRequest<AssetRecord>(`/api/assets/${id}`);
