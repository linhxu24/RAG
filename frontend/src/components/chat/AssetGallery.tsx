import { useState } from "react";
import { ImageOff } from "lucide-react";

import type { AssetRecord } from "../../types";
import { assetUrl } from "../../utils/format";

export function AssetGallery({ assets }: { assets: AssetRecord[] }) {
  if (!assets.length) return null;
  return (
    <div className="mt-4 grid grid-cols-2 gap-3">
      {assets.map((asset) => (
        <AssetCard key={asset.asset_id || asset.url} asset={asset} />
      ))}
    </div>
  );
}

function AssetCard({ asset }: { asset: AssetRecord }) {
  const [broken, setBroken] = useState(false);
  const source = assetUrl(asset.url || asset.public_url);
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      {source && !broken ? (
        <img
          src={source}
          alt={asset.asset_token || asset.asset_id}
          className="h-40 w-full object-cover"
          onError={() => setBroken(true)}
        />
      ) : (
        <div className="flex h-40 flex-col items-center justify-center gap-2 bg-slate-100 text-slate-500">
          <ImageOff size={24} />
          <span className="text-xs font-semibold">Broken asset</span>
        </div>
      )}
      <div className="p-2.5">
        <div className="mono truncate text-[10px] text-slate-500">{asset.asset_id}</div>
        <div className="mt-1 text-xs font-semibold text-slate-700">
          {asset.type || asset.asset_type || "image"}
        </div>
      </div>
    </div>
  );
}
