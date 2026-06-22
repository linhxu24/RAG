import {
  Activity,
  Bot,
  Database,
  FileSearch,
  Gauge,
  Image,
  SearchCode,
  Settings,
  SquareStack,
  UploadCloud,
  type LucideIcon,
} from "lucide-react";

export interface NavigationItem {
  to: string;
  label: string;
  shortLabel: string;
  description: string;
  icon: LucideIcon;
}

export const navigationGroups: Array<{
  label: string;
  items: NavigationItem[];
}> = [
  {
    label: "Workspace",
    items: [
      {
        to: "/chatbot",
        label: "Chatbot",
        shortLabel: "Chatbot",
        description: "Test hội thoại và FAQ",
        icon: Bot,
      },
      {
        to: "/upload",
        label: "Upload Documents",
        shortLabel: "Upload",
        description: "Nạp tài liệu và ảnh đi kèm",
        icon: UploadCloud,
      },
      {
        to: "/documents",
        label: "Document Store",
        shortLabel: "Documents",
        description: "Quản lý tài liệu đã ingest",
        icon: SquareStack,
      },
    ],
  },
  {
    label: "Pipeline",
    items: [
      {
        to: "/ingestion",
        label: "Ingestion Monitor",
        shortLabel: "Ingestion",
        description: "Theo dõi parsing và quality checks",
        icon: Activity,
      },
      {
        to: "/retrieval",
        label: "Retrieval Playground",
        shortLabel: "Retrieval",
        description: "Kiểm tra structured và hybrid retrieval",
        icon: SearchCode,
      },
      {
        to: "/evaluation",
        label: "Evaluation Dashboard",
        shortLabel: "Evaluation",
        description: "Đo chất lượng pipeline",
        icon: Gauge,
      },
    ],
  },
  {
    label: "Operations",
    items: [
      {
        to: "/observability",
        label: "Observability",
        shortLabel: "Observability",
        description: "Health, latency và cảnh báo",
        icon: Activity,
      },
      {
        to: "/traces",
        label: "Trace Explorer",
        shortLabel: "Traces",
        description: "Điều tra từng request",
        icon: FileSearch,
      },
      {
        to: "/assets",
        label: "Asset Manager",
        shortLabel: "Assets",
        description: "Kiểm tra ảnh và asset token",
        icon: Image,
      },
      {
        to: "/data",
        label: "Data Tables",
        shortLabel: "Data",
        description: "Xem dữ liệu nghiệp vụ",
        icon: Database,
      },
      {
        to: "/settings",
        label: "Settings",
        shortLabel: "Settings",
        description: "Xem cấu hình đang chạy",
        icon: Settings,
      },
    ],
  },
];

export const navigationItems = navigationGroups.flatMap((group) => group.items);

export function navigationItemForPath(pathname: string): NavigationItem {
  return (
    navigationItems.find(
      (item) => pathname === item.to || pathname.startsWith(`${item.to}/`),
    ) ?? navigationItems[0]
  );
}
