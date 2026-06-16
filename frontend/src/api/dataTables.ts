import type {
  ApiList,
  ProductRecord,
  ServiceRecord,
} from "../types";
import { apiRequest } from "./client";

export type DataTableName =
  | "products"
  | "services"
  | "faqs"
  | "clinic-info"
  | "tables"
  | "table-rows"
  | "chunks";

export const getProducts = () =>
  apiRequest<ApiList<ProductRecord>>("/api/products");
export const getServices = () =>
  apiRequest<ApiList<ServiceRecord>>("/api/services");
export const getDataTable = <T = Record<string, any>>(name: DataTableName) =>
  apiRequest<ApiList<T>>(`/api/${name}`);
