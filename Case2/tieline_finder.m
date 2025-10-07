function [LD_out, TL_out] = tieline_finder(BD,LD,TL)
%%% MST to ensure connectivity, used as input for reconfiguration_func
    buses = BD(:,1);         
    numBuses = length(buses); 
    % For each row i in LD, columns:
    %  - LD(i,2) = From Bus
    %  - LD(i,3) = To Bus
    %  - LD(i,4) = R (ohms) -> weight
    edgesLD = [LD(:,2), LD(:,3), LD(:,4)];
    edgesTL = [TL(:,2), TL(:,3), TL(:,4)];
    % Format: [fromBus, toBus, resistance]
    allEdges = [edgesLD; edgesTL];

    % Create an undirected graph using the bus numbers as node IDs
    G = graph(allEdges(:,1), allEdges(:,2), allEdges(:,3));
    MST = minspantree(G);

    
    % MST.Edges table has two columns for the endpoints:
    mstPairs = MST.Edges.EndNodes;  % Each row is [u, v]
    
    % --- 3. Identify Which Input Edges Were Used in the MST ---
    % For matching (order does not matter) we sort the node pairs.
    LD_pairs = sort(LD(:,2:3), 2);
    TL_pairs = sort(TL(:,2:3), 2);
    
    % Initialize boolean masks for LD and TL edges.
    usedLD = false(size(LD,1), 1);
    usedTL = false(size(TL,1), 1);
    
    % Loop over each MST edge (there should be 33-1 = 32 edges)
    for i = 1:size(mstPairs,1)
        edge = sort(mstPairs(i,:));  % sort to ignore order
        % First, check if this edge is in LD
        idxLD = find( (LD_pairs(:,1) == edge(1)) & (LD_pairs(:,2) == edge(2)) );
        if ~isempty(idxLD)
            usedLD(idxLD) = true;
        else
            % Otherwise, check in TL
            idxTL = find( (TL_pairs(:,1) == edge(1)) & (TL_pairs(:,2) == edge(2)) );
            if ~isempty(idxTL)
                usedTL(idxTL) = true;
            else
                warning('Edge [%d, %d] not found in LD or TL', edge(1), edge(2));
            end
        end
    end

    % --- 4. Partition the Input Edges into Active (used) and Inactive (unused) ---
    % Active edges: those from LD and TL that were used in the MST.
    LD_out = [LD(usedLD, :); TL(usedTL, :)];
    
    % Inactive edges: the ones from LD that were not used and
    % the ones from TL that were not used.
    TL_out = [LD(~usedLD, :); TL(~usedTL, :)];
    
    % At this point, note:
    % Total input edges = size(LD,1) + size(TL,1) = 32 + 5 = 37.
    % MST uses 32 edges, so TL_out should have 37 - 32 = 5 rows.

    
    % Display edges of MST
    %disp('Edges in the MST (From, To, Resistance):'); 
    % figure;
    % plot(MST,'Layout','force');
    % title('33-Bus System Graph MST (Edges Weighted by Resistance)');
end

