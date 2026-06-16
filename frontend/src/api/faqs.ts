import { apiRequest } from "./client";

export interface PublicFaq {
  faq_id: string;
  question: string;
  answer: string;
  category?: string | null;
  category_code?: string | null;
  keywords: string[];
}

export const listPublicFaqs = (search = "", category = "") => {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  if (category) params.set("category", category);
  const query = params.size ? `?${params.toString()}` : "";
  return apiRequest<{ items: PublicFaq[]; count: number }>(`/faqs${query}`);
};
