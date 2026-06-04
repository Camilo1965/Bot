interface Props {
  icon?: string;
  title: string;
  body?: string;
  action?: { label: string; onClick: () => void };
}

export function EmptyState({ icon = "○", title, body, action }: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <div className="w-12 h-12 rounded-full bg-[#1E2530] border border-[#374151] flex items-center justify-center text-[#9CA3AF] text-xl mb-3">
        {icon}
      </div>
      <h3 className="text-sm font-medium text-[#F3F4F6] mb-1">{title}</h3>
      {body && <p className="text-xs text-[#9CA3AF] max-w-sm">{body}</p>}
      {action && (
        <button
          onClick={action.onClick}
          className="mt-4 px-3 py-1.5 rounded text-xs font-medium bg-[#3B82F6]/20 text-[#3B82F6] hover:bg-[#3B82F6]/30 transition-colors"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
